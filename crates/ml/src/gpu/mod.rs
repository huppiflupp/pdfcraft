//! The wgpu device and what runs on it.
//!
//! Every recorded piece of work runs in [`Gpu::scoped`]: validation and out-of-memory errors are
//! caught in wgpu error scopes and come back as [`GpuError`], as do uncaptured errors and a lost
//! device (wgpu's default handler would panic). Buffer sizes and dispatch sizes are checked
//! against the device's limits before anything is created.

mod detector;
mod graph;
mod nd;
mod pp;
mod recognizer;
pub(crate) mod shaders;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub use detector::Detector;
pub use pp::{PpRecognizer, Step};
pub use recognizer::Recognizer;

use crate::onnx::OnnxError;

pub(crate) type Buf = Arc<wgpu::Buffer>;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no usable GPU: {0}")]
    NoAdapter(String),
    #[error("GPU: {0}")]
    Run(String),
    #[error("model: {0}")]
    Model(String),
    #[error("too large for this GPU: {0}")]
    TooLarge(String),
}

impl From<OnnxError> for GpuError {
    fn from(e: OnnxError) -> Self {
        GpuError::Model(e.to_string())
    }
}

/// Per-device choices, from measurements on an RTX 5080 and a Radeon 8060S (RDNA 3.5): the
/// 128-pixel convolution tile pays on NVIDIA and is slower on RDNA; the GRU recurrence wants 4
/// lines per workgroup on the former and 8 on the latter.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tuning {
    pub conv128: bool,
    pub lines: usize,
    pub unroll: usize,
    pub f16: bool,
}

/// Bytes of scratch buffers kept for reuse (wgpu zero-fills every new buffer, so reusing them
/// saves that); beyond this, freed buffers are dropped.
const POOL_CAP: u64 = 512 << 20;

pub struct Gpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    adapter: String,
    pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    limits: wgpu::Limits,
    pool: Mutex<(HashMap<u64, Vec<Buf>>, u64)>,
    pub(crate) tuning: Tuning,
    /// An uncaptured error or the device's loss, reported by wgpu's callbacks.
    failure: Arc<Mutex<Option<String>>>,
}

impl Gpu {
    /// The high-performance adapter, if it is a real GPU (a software rasteriser would be slower
    /// than the CPU path), with the pipelines compiled.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }),
        )
        .map_err(|e| GpuError::NoAdapter(e.to_string()))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(GpuError::NoAdapter(format!("{} is a software rasteriser", info.name)));
        }
        let f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: if f16 { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() },
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| GpuError::NoAdapter(e.to_string()))?;
        let failure: Arc<Mutex<Option<String>>> = Arc::default();
        let f = failure.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            if let Ok(mut s) = f.lock() {
                s.get_or_insert_with(|| e.to_string());
            }
        }));
        let f = failure.clone();
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut s) = f.lock() {
                s.get_or_insert_with(|| format!("device lost ({reason:?}): {message}"));
            }
        });
        let nvidia = info.vendor == 0x10de;
        let tuning = Tuning { conv128: nvidia, lines: if nvidia { 4 } else { 8 }, unroll: 4, f16 };
        let mut gpu = Self {
            device,
            queue,
            adapter: format!("{} ({:?})", info.name, info.backend),
            pipes: HashMap::new(),
            limits,
            pool: Mutex::default(),
            tuning,
            failure,
        };
        gpu.pipes = gpu.scoped(|| Ok(gpu.compile()))?;
        Ok(gpu)
    }

    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    fn compile(&self) -> HashMap<&'static str, wgpu::ComputePipeline> {
        use shaders::*;
        let rec = recurrence(self.tuning.lines, self.tuning.f16, self.tuning.unroll);
        let rec = if self.tuning.f16 {
            format!("enable f16;\n{}", rec.replace("var<storage, read> r: array<vec4<f32>>", "var<storage, read> r: array<vec4<f16>>"))
        } else {
            rec
        };
        let sources: Vec<(&'static str, String)> = vec![
            ("conv_fast", shader(&CONV.replace("FAST_OR_GENERIC", CONV_FAST_LOAD))),
            ("conv", shader(&CONV.replace("FAST_OR_GENERIC", CONV_GENERIC_LOAD))),
            ("conv128", shader(CONV128)),
            ("conv_direct", shader(CONV_DIRECT)),
            ("depthwise", shader(DEPTHWISE)),
            ("convt", shader(CONV_TRANSPOSE)),
            ("maxpool", shader(MAXPOOL)),
            ("slice", shader(SLICE)),
            ("concat", shader(CONCAT)),
            ("activate", shader(ACTIVATE)),
            ("avgpool_t", shader(AVGPOOL_T)),
            ("grey_resize", shader(GREY_RESIZE)),
            ("mask_up", shader(MASK_UP)),
            ("gemm", GEMM.to_string()),
            ("recurrence", rec),
            ("nd_binary", shader(nd::BINARY)),
            ("nd_unary", shader(nd::UNARY)),
            ("nd_reduce_mean", shader(nd::REDUCE_MEAN)),
            ("nd_reduce_mean_wide", nd::REDUCE_MEAN_WIDE.to_string()),
            ("nd_layer_norm", shader(nd::LAYER_NORM)),
            ("nd_softmax", shader(nd::SOFTMAX)),
            ("nd_copy", shader(nd::COPY)),
            ("nd_pool", shader(nd::POOL)),
            ("nd_affine", shader(nd::CHANNEL_AFFINE)),
            ("nd_bgemm", shader(nd::BGEMM)),
            ("softmax_topk", nd::SOFTMAX_TOPK.to_string()),
        ];
        let mut pipes = HashMap::new();
        for (name, src) in sources {
            let module =
                self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) });
            let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            pipes.insert(name, pipe);
        }
        pipes
    }

    /// Run `f` (recording, submitting, reading back) with wgpu's validation and out-of-memory
    /// errors caught and returned, as is any uncaptured error or device loss meanwhile.
    pub(crate) fn scoped<T>(&self, f: impl FnOnce() -> Result<T, GpuError>) -> Result<T, GpuError> {
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let result = f();
        let caught = pollster::block_on(memory.pop()).or(pollster::block_on(validation.pop()));
        if let Some(e) = caught {
            return Err(GpuError::Run(e.to_string()));
        }
        if let Some(e) = self.failure.lock().ok().and_then(|f| f.clone()) {
            return Err(GpuError::Run(e));
        }
        result
    }

    /// [`Self::storage`] for at least `floats` values, rounded up to one of four sizes per power
    /// of two (at most a quarter more), so that buffers freed while a graph is being recorded fit
    /// later values of a similar size: the pool only hands out exact sizes, and a buffer dropped
    /// before its work is submitted still holds its memory until that work is done.
    pub(crate) fn storage_rounded(&self, floats: usize) -> Result<Buf, GpuError> {
        self.storage(rounded(floats).ok_or_else(|| GpuError::TooLarge(format!("{floats} values")))?)
    }

    /// A scratch buffer of `floats` 32-bit values, reused from the pool when one fits exactly.
    pub(crate) fn storage(&self, floats: usize) -> Result<Buf, GpuError> {
        let size = floats.max(1).checked_mul(4).map(|s| s as u64).ok_or_else(|| GpuError::TooLarge(format!("{floats} values")))?;
        if size > self.limits.max_buffer_size || size > self.limits.max_storage_buffer_binding_size {
            return Err(GpuError::TooLarge(format!("a {size}-byte buffer")));
        }
        if let Ok(mut pool) = self.pool.lock()
            && let Some(b) = pool.0.get_mut(&size).and_then(Vec::pop)
        {
            pool.1 = pool.1.saturating_sub(size);
            return Ok(b);
        }
        Ok(Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })))
    }

    /// Give scratch buffers back once the work using them is recorded (wgpu orders later use
    /// after it). Only buffers made by [`Self::storage`] are kept, up to [`POOL_CAP`]: a full pool
    /// drops buffers of other sizes first, since the sizes still in use are the ones just freed
    /// (each batch of lines has its own width, so its own sizes).
    pub(crate) fn recycle(&self, bufs: impl IntoIterator<Item = Buf>) {
        let Ok(mut pool) = self.pool.lock() else { return };
        let scratch = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        for b in bufs {
            if b.usage() != scratch || b.size() > POOL_CAP {
                continue;
            }
            while pool.1.saturating_add(b.size()) > POOL_CAP {
                let other = pool.0.iter().find(|(s, v)| **s != b.size() && !v.is_empty()).map(|(s, _)| *s);
                let Some(size) = other.or_else(|| pool.0.iter().find(|(_, v)| !v.is_empty()).map(|(s, _)| *s)) else { break };
                let evicted = pool.0.get_mut(&size).and_then(Vec::pop);
                pool.1 = pool.1.saturating_sub(evicted.map_or(0, |e| e.size()));
                if pool.0.get(&size).is_some_and(Vec::is_empty) {
                    pool.0.remove(&size);
                }
            }
            if pool.1.saturating_add(b.size()) > POOL_CAP {
                continue;
            }
            pool.1 += b.size();
            pool.0.entry(b.size()).or_default().push(b);
        }
    }

    fn init(&self, bytes: &[u8], usage: wgpu::BufferUsages) -> Result<Buf, GpuError> {
        let len = bytes.len().max(4) as u64;
        if len > self.limits.max_buffer_size || len > self.limits.max_storage_buffer_binding_size {
            return Err(GpuError::TooLarge(format!("a {len}-byte buffer")));
        }
        use wgpu::util::DeviceExt;
        let zero = [0u8; 4];
        Ok(Arc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: if bytes.is_empty() { &zero } else { bytes },
            usage,
        })))
    }

    pub(crate) fn upload(&self, data: &[f32]) -> Result<Buf, GpuError> {
        self.init(bytemuck::cast_slice(data), wgpu::BufferUsages::STORAGE)
    }

    pub(crate) fn upload_u32(&self, data: &[u32]) -> Result<Buf, GpuError> {
        self.init(bytemuck::cast_slice(data), wgpu::BufferUsages::STORAGE)
    }

    pub(crate) fn upload_bytes(&self, data: &[u8]) -> Result<Buf, GpuError> {
        self.init(data, wgpu::BufferUsages::STORAGE)
    }

    /// A uniform buffer of `v`, padded to whole 16-byte rows.
    pub(crate) fn uniform(&self, v: &[u32]) -> Result<Buf, GpuError> {
        let mut v = v.to_vec();
        while !v.len().is_multiple_of(4) {
            v.push(0);
        }
        self.init(bytemuck::cast_slice(&v), wgpu::BufferUsages::UNIFORM)
    }

    pub(crate) fn encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&Default::default())
    }

    pub(crate) fn dispatch(&self, enc: &mut wgpu::CommandEncoder, pipe: &str, buffers: &[&Buf], groups: (u32, u32, u32)) -> Result<(), GpuError> {
        let max = self.limits.max_compute_workgroups_per_dimension;
        if groups.0 > max || groups.1 > max || groups.2 > max {
            return Err(GpuError::TooLarge(format!("{groups:?} workgroups")));
        }
        if groups.0 == 0 || groups.1 == 0 || groups.2 == 0 {
            return Ok(());
        }
        let pipe = self.pipes.get(pipe).ok_or_else(|| GpuError::Run(format!("no pipeline {pipe}")))?;
        let entries: Vec<wgpu::BindGroupEntry> =
            buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let bind =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        Ok(())
    }

    /// One thread per element over a 2D grid of 256-wide workgroups.
    pub(crate) fn elementwise(&self, enc: &mut wgpu::CommandEncoder, pipe: &str, buffers: &[&Buf], total: usize) -> Result<(), GpuError> {
        let max = self.limits.max_compute_workgroups_per_dimension as usize;
        let g = total.div_ceil(256);
        let (x, y) = (g.min(max), g.div_ceil(max.max(1)));
        self.dispatch(enc, pipe, buffers, (to_u32(x)?, to_u32(y)?, 1))
    }

    /// A workgroup per row over `rows` rows, on a 2D grid (row = y * width + x).
    pub(crate) fn rows(&self, enc: &mut wgpu::CommandEncoder, pipe: &str, buffers: &[&Buf], rows: usize) -> Result<(), GpuError> {
        let max = self.limits.max_compute_workgroups_per_dimension as usize;
        let (x, y) = (rows.min(max), rows.div_ceil(max.max(1)));
        self.dispatch(enc, pipe, buffers, (to_u32(x)?, to_u32(y)?, 1))
    }

    /// Submit `enc` with a copy of the first `count` 32-bit values of `buf`, wait, and return
    /// them.
    pub(crate) fn read(&self, mut enc: wgpu::CommandEncoder, buf: &Buf, count: usize) -> Result<Vec<u32>, GpuError> {
        let size = (count.max(1) as u64).checked_mul(4).ok_or_else(|| GpuError::TooLarge(format!("{count} values")))?;
        if size > buf.size() {
            return Err(GpuError::Run(format!("reading {size} bytes of a {}-byte buffer", buf.size())));
        }
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(buf, 0, &read, 0, size);
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| GpuError::Run(e.to_string()))?;
        rx.recv().map_err(|e| GpuError::Run(e.to_string()))?.map_err(|e| GpuError::Run(e.to_string()))?;
        let view = read.slice(..).get_mapped_range().map_err(|e| GpuError::Run(e.to_string()))?;
        let values = bytemuck::try_cast_slice::<u8, u32>(&view[..]).map_err(|e| GpuError::Run(e.to_string()))?.to_vec();
        drop(view);
        Ok(values)
    }

    /// Submit `enc` and wait for it.
    pub(crate) fn submit_wait(&self, enc: wgpu::CommandEncoder) -> Result<(), GpuError> {
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).map(|_| ()).map_err(|e| GpuError::Run(e.to_string()))
    }
}

/// `n` rounded up to 1, 1.25, 1.5 or 1.75 times a power of two (powers of two up to 1024).
fn rounded(n: usize) -> Option<usize> {
    let n = n.max(1);
    if n <= 1024 {
        return n.checked_next_power_of_two();
    }
    let step = (1usize << n.ilog2()) / 4;
    n.div_ceil(step).checked_mul(step)
}

pub(crate) fn to_u32(v: usize) -> Result<u32, GpuError> {
    u32::try_from(v).map_err(|_| GpuError::TooLarge(format!("{v} does not fit 32 bits")))
}

/// `v` as u32 values for a uniform or parameter buffer.
pub(crate) fn params(v: &[usize]) -> Result<Vec<u32>, GpuError> {
    v.iter().map(|&x| to_u32(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::rounded;

    #[test]
    fn rounded_sizes_waste_at_most_a_quarter() {
        assert_eq!(rounded(0), Some(1));
        assert_eq!(rounded(1000), Some(1024));
        assert_eq!(rounded(1025), Some(1280));
        assert_eq!(rounded(1280), Some(1280));
        assert_eq!(rounded(1500), Some(1536));
        assert_eq!(rounded(2047), Some(2048));
        for n in [1025usize, 3000, 77_777, 123_456_789] {
            let r = rounded(n).unwrap_or(0);
            assert!(r >= n && r * 4 <= n * 5 + 4, "{n} -> {r}");
        }
        assert_eq!(rounded(usize::MAX), None);
    }
}
