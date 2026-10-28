//! An ONNX graph executor for N-dimensional float tensors, enough for the PP-OCR recognition
//! models (a CNN backbone, the SVTR transformer neck, the CTC head): PP-OCRv6 and the PP-OCRv5
//! Latin model. Ported from the wgpu-ocr prototype (github.com/huppiflupp/wgpu-ocr).
//!
//! Shapes are resolved for each input as the graph runs: the integer "shape" subgraphs (Shape,
//! Gather, Slice, Squeeze, Unsqueeze, Concat, Cast, integer arithmetic) are computed on the CPU,
//! float nodes become dispatches. Identity nodes are removed up front. A constant bias Add and
//! an activation (Relu, HardSwish, GELU) after a Conv, a residual Add after it, and a
//! squeeze-excitation scale before it are fused into the convolution; a bias and SiLU into a
//! MatMul with a constant weight; BatchNormalization becomes a per-channel scale and shift; the
//! exporter's nine-node LayerNorm one kernel. The graph's final Softmax is left to the caller
//! ([`super::PpRecognizer`] fuses it with the ranking).

use std::collections::{HashMap, HashSet};

use super::graph::count;
use super::{Buf, Gpu, GpuError, params, to_u32};
use crate::onnx::{Node, Onnx};

/// The most dimensions a tensor may have here (the copy and broadcast kernels' parameter
/// layout).
const RANK: usize = 6;

/// Element-wise binary operation with broadcasting: y = a op b.
pub(crate) const BINARY: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;  // total, rank, op, 0, dims[6], astr[6], bstr[6]
ENTRY
    INDEX
    if (i >= p[0]) { return; }
    var rem = i;
    var ai = 0u;
    var bi = 0u;
    for (var k = 0u; k < p[1]; k++) {
        let d = p[1] - 1u - k;
        let dim = p[4u + d];
        let c = rem % dim;
        rem = rem / dim;
        ai += c * p[10u + d];
        bi += c * p[16u + d];
    }
    let x = a[ai];
    let z = b[bi];
    var v = 0.0;
    switch p[2] {
        case 0u: { v = x + z; }
        case 1u: { v = x - z; }
        case 2u: { v = x * z; }
        case 3u: { v = x / z; }
        default: {
            if (z == 2.0) { v = x * x; } else if (z == 0.5) { v = sqrt(x); } else { v = pow(x, z); }
        }
    }
    y[i] = v;
}
"#;

pub(crate) const UNARY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;  // total, op, alpha bits, beta bits
EPILOGUE_FN
ENTRY
    INDEX
    if (i >= p.x) { return; }
    let v = x[i];
    var r = v;
    switch p.y {
        case 1u: { r = max(v, 0.0); }
        case 2u: { r = 1.0 / (1.0 + exp(-v)); }
        case 3u: { r = clamp(bitcast<f32>(p.z) * v + bitcast<f32>(p.w), 0.0, 1.0); }
        case 4u: { r = erf_approx(v); }
        case 5u: { r = sqrt(v); }
        case 6u: { r = activate(v, 5u); }
        default: {}
    }
    y[i] = r;
}
"#;

/// Mean over the trailing `inner` values of each of `outer` rows, a thread per row.
pub(crate) const REDUCE_MEAN: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;  // outer, inner
ENTRY
    INDEX
    if (i >= p.x) { return; }
    var s = 0.0;
    for (var j = 0u; j < p.y; j++) { s += x[i * p.y + j]; }
    y[i] = s / f32(p.y);
}
"#;

/// The same with a workgroup per row, for long rows (squeeze-excitation over H x W).
pub(crate) const REDUCE_MEAN_WIDE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;  // outer, inner
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let row = wid.y * nw.x + wid.x;
    var s = 0.0;
    if (row < p.x) {
        for (var j = lid.x; j < p.y; j += 256u) { s += x[row * p.y + j]; }
    }
    part[lid.x] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k / 2u) {
        if (lid.x < k) { part[lid.x] += part[lid.x + k]; }
        workgroupBarrier();
    }
    if (lid.x == 0u && row < p.x) { y[row] = part[0] / f32(p.y); }
}
"#;

/// LayerNorm over the last axis: (x - mean) / sqrt(var + eps) * gamma + beta, a thread per row.
pub(crate) const LAYER_NORM: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gb: array<f32>;  // gamma[len] then beta[len]
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;  // rows, len, eps bits
ENTRY
    INDEX
    if (i >= p.x) { return; }
    let base = i * p.y;
    var m = 0.0;
    for (var j = 0u; j < p.y; j++) { m += x[base + j]; }
    m = m / f32(p.y);
    var v = 0.0;
    for (var j = 0u; j < p.y; j++) { let d = x[base + j] - m; v += d * d; }
    let inv = 1.0 / sqrt(v / f32(p.y) + bitcast<f32>(p.z));
    for (var j = 0u; j < p.y; j++) { y[base + j] = (x[base + j] - m) * inv * gb[j] + gb[p.y + j]; }
}
"#;

/// Softmax over the last `len` values of each row, a thread per row (the attention inside the
/// neck; the graph's final Softmax is fused into the ranking instead).
pub(crate) const SOFTMAX: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;  // rows, row length
ENTRY
    INDEX
    if (i >= p.x) { return; }
    let base = i * p.y;
    var m = -3.4e38;
    for (var j = 0u; j < p.y; j++) { m = max(m, x[base + j]); }
    var s = 0.0;
    for (var j = 0u; j < p.y; j++) { s += exp(x[base + j] - m); }
    for (var j = 0u; j < p.y; j++) { y[base + j] = exp(x[base + j] - m) / s; }
}
"#;

/// y[obase + sum c_d os_d] = x[ibase + sum c_d is_d] over the index space `dims`: Transpose,
/// Slice and Concat are all this.
pub(crate) const COPY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;  // total, rank, ibase, obase, dims[6], is[6], os[6]
ENTRY
    INDEX
    if (i >= p[0]) { return; }
    var rem = i;
    var xi = p[2];
    var yi = p[3];
    for (var k = 0u; k < p[1]; k++) {
        let d = p[1] - 1u - k;
        let dim = p[4u + d];
        let c = rem % dim;
        rem = rem / dim;
        xi += c * p[10u + d];
        yi += c * p[16u + d];
    }
    y[yi] = x[xi];
}
"#;

/// Max or average pooling with stride and leading padding (padding never counts).
pub(crate) const POOL: &str = r#"
struct P { nc: u32, h: u32, w: u32, oh: u32, ow: u32, kh: u32, kw: u32, sh: u32, sw: u32, ph: u32, pw: u32, avg: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
ENTRY
    INDEX
    if (i >= p.nc * p.oh * p.ow) { return; }
    let ox = i % p.ow;
    let oy = (i / p.ow) % p.oh;
    let c = i / (p.ow * p.oh);
    var m = -3.4e38;
    var s = 0.0;
    var n = 0.0;
    for (var ky = 0u; ky < p.kh; ky++) {
        let iy = i32(oy * p.sh + ky) - i32(p.ph);
        if (iy < 0 || iy >= i32(p.h)) { continue; }
        for (var kx = 0u; kx < p.kw; kx++) {
            let ix = i32(ox * p.sw + kx) - i32(p.pw);
            if (ix < 0 || ix >= i32(p.w)) { continue; }
            let v = x[(c * p.h + u32(iy)) * p.w + u32(ix)];
            m = max(m, v);
            s += v;
            n += 1.0;
        }
    }
    if (p.avg == 1u) { y[i] = s / max(n, 1.0); } else { y[i] = m; }
}
"#;

/// y = x * scale[c] + shift[c], c the axis-1 index (BatchNormalization folded).
pub(crate) const CHANNEL_AFFINE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> st: array<f32>;  // scale[c] then shift[c]
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;  // total, channels, inner
ENTRY
    INDEX
    if (i >= p.x) { return; }
    let c = (i / p.z) % p.y;
    y[i] = x[i] * st[c] + st[p.y + c];
}
"#;

/// Batched y[b][m, k] = act(x[b][m, :] . w[b][:, k] + bias[k]), batch strides in elements (0 for
/// a shared weight). 64 x 64 tile, 4 x 4 per thread.
pub(crate) const BGEMM: &str = r#"
struct D { m: u32, k: u32, i: u32, act: u32, xs: u32, ws: u32, ys: u32, x0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> d: D;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
EPILOGUE_FN

fn put(base: u32, m: u32, c0: u32, v: vec4<f32>) {
    if (m >= d.m) { return; }
    for (var j = 0u; j < 4u; j++) {
        let c = c0 + j;
        if (c < d.k) { y[base + m * d.k + c] = activate(v[j] + b[c], d.act); }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.y * 16u + lid.x;
    let m0 = wid.x * 64u;
    let c0 = wid.y * 64u;
    let xb = wid.z * d.xs;
    let wb = wid.z * d.ws;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < d.i; k0 += 16u) {
        for (var e = 0u; e < 4u; e++) {
            let idx = tid + e * 256u;
            let am = idx / 16u;
            let ak = idx % 16u;
            var v = 0.0;
            if (m0 + am < d.m && k0 + ak < d.i) { v = x[xb + (m0 + am) * d.i + k0 + ak]; }
            As[ak * 64u + am] = v;
            let bk = idx / 64u;
            let bc = idx % 64u;
            var wv = 0.0;
            if (k0 + bk < d.i && c0 + bc < d.k) { wv = wt[wb + (k0 + bk) * d.k + c0 + bc]; }
            Bs[bk * 64u + bc] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let ab = kk * 64u + lid.x * 4u;
            let a = vec4<f32>(As[ab], As[ab + 1u], As[ab + 2u], As[ab + 3u]);
            let bb = kk * 64u + lid.y * 4u;
            let bv = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
            acc0 += a.x * bv;
            acc1 += a.y * bv;
            acc2 += a.z * bv;
            acc3 += a.w * bv;
        }
        workgroupBarrier();
    }
    let base = wid.z * d.ys;
    let mb = m0 + lid.x * 4u;
    let cb = c0 + lid.y * 4u;
    put(base, mb, cb, acc0);
    put(base, mb + 1u, cb, acc1);
    put(base, mb + 2u, cb, acc2);
    put(base, mb + 3u, cb, acc3);
}
"#;

/// The final Softmax and the ranking in one: for each row of logits, the probability (softmax
/// over every class) of the [`super::pp::TOP`] classes with the highest logits among those the
/// mask allows, best first; a workgroup of 128 per row. Out: per row TOP class indices, then TOP
/// probability bits; an empty slot has the index 0xffffffff.
pub(crate) const SOFTMAX_TOPK: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;  // a bit per class
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;  // rows, classes, masked
const T = 8u;
const NONE = 0xffffffffu;
var<workgroup> red: array<f32, 128>;
var<workgroup> tv: array<f32, 1024>;
var<workgroup> ti: array<u32, 1024>;
fn allowed(c: u32) -> bool { return p.z == 0u || ((mask[c / 32u] >> (c % 32u)) & 1u) == 1u; }
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let row = wid.y * nw.x + wid.x;
    let t = lid.x;
    let live = row < p.x;
    let base = row * p.y;
    var m = -3.4e38;
    if (live) { for (var j = t; j < p.y; j += 128u) { m = max(m, x[base + j]); } }
    red[t] = m;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s / 2u) {
        if (t < s) { red[t] = max(red[t], red[t + s]); }
        workgroupBarrier();
    }
    let top = red[0];
    workgroupBarrier();
    var sum = 0.0;
    var lv: array<f32, 8>;
    var li: array<u32, 8>;
    for (var k = 0u; k < T; k++) { lv[k] = -3.4e38; li[k] = NONE; }
    if (live) {
        for (var j = t; j < p.y; j += 128u) {
            let v = x[base + j];
            sum += exp(v - top);
            if (allowed(j) && (li[T - 1u] == NONE || v > lv[T - 1u])) {
                var k = T - 1u;
                loop {
                    if (k == 0u) { break; }
                    if (li[k - 1u] != NONE && lv[k - 1u] >= v) { break; }
                    lv[k] = lv[k - 1u];
                    li[k] = li[k - 1u];
                    k -= 1u;
                }
                lv[k] = v;
                li[k] = j;
            }
        }
    }
    red[t] = sum;
    for (var k = 0u; k < T; k++) { tv[t * T + k] = lv[k]; ti[t * T + k] = li[k]; }
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s / 2u) {
        if (t < s) {
            red[t] += red[t + s];
            var ov: array<f32, 8>;
            var oi: array<u32, 8>;
            var a = 0u;
            var b = 0u;
            for (var k = 0u; k < T; k++) {
                let av = tv[t * T + a];
                let ai = ti[t * T + a];
                let bv = tv[(t + s) * T + b];
                let bi = ti[(t + s) * T + b];
                let take_a = bi == NONE || (ai != NONE && (av > bv || (av == bv && ai < bi)));
                if (take_a) { ov[k] = av; oi[k] = ai; a += 1u; } else { ov[k] = bv; oi[k] = bi; b += 1u; }
            }
            for (var k = 0u; k < T; k++) { tv[t * T + k] = ov[k]; ti[t * T + k] = oi[k]; }
        }
        workgroupBarrier();
    }
    if (t == 0u && live) {
        let total = red[0];
        for (var k = 0u; k < T; k++) {
            let c = ti[k];
            out[row * 2u * T + k] = c;
            var prob = 0.0;
            if (c != NONE) { prob = exp(tv[k] - top) / total; }
            out[row * 2u * T + T + k] = bitcast<u32>(prob);
        }
    }
}
"#;

/// A float tensor on the GPU, any rank up to [`RANK`].
#[derive(Clone)]
pub(crate) struct Tensor {
    pub buf: Buf,
    pub shape: Vec<usize>,
}

#[derive(Clone)]
enum Value {
    F(Tensor),
    /// An integer tensor kept on the CPU (shapes and indices), with its own shape.
    I(Vec<i64>, Vec<usize>),
}

fn model(m: impl Into<String>) -> GpuError {
    GpuError::Model(m.into())
}

fn dim(shape: &[usize], d: usize) -> Result<usize, GpuError> {
    shape.get(d).copied().ok_or_else(|| model(format!("no axis {d} in {shape:?}")))
}

fn strides(shape: &[usize]) -> Result<Vec<usize>, GpuError> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        let next = s.get(d + 1).copied().unwrap_or(1).checked_mul(dim(shape, d + 1)?).ok_or_else(|| GpuError::TooLarge(format!("{shape:?}")))?;
        if let Some(x) = s.get_mut(d) {
            *x = next;
        }
    }
    Ok(s)
}

fn axis(a: i64, rank: usize) -> Result<usize, GpuError> {
    let r = if a < 0 { a.checked_add(rank as i64) } else { Some(a) };
    match r.and_then(|r| usize::try_from(r).ok()) {
        Some(r) if r < rank.max(1) => Ok(r),
        _ => Err(model(format!("axis {a} out of range for rank {rank}"))),
    }
}

fn dim_of(v: i64) -> Result<usize, GpuError> {
    usize::try_from(v).map_err(|_| model(format!("negative size {v}")))
}

/// The output length of a convolution or pool: (size + pads - k) / stride + 1.
fn out_len(size: usize, pads: usize, k: usize, stride: usize) -> Result<usize, GpuError> {
    let span = size.checked_add(pads).and_then(|s| s.checked_sub(k)).ok_or_else(|| model(format!("kernel {k} larger than input {size}")))?;
    Ok(span / stride.max(1) + 1)
}

/// Leading and trailing padding along one axis for SAME_UPPER (the odd one at the end).
fn same_upper(size: usize, k: usize, stride: usize) -> (usize, usize) {
    let out = size.div_ceil(stride.max(1));
    let total = out.saturating_sub(1).saturating_mul(stride).saturating_add(k).saturating_sub(size);
    (total / 2, total - total / 2)
}

/// Four leading/trailing pads [top, left, bottom, right] from an ONNX `pads` attribute.
fn pads4(p: Option<Vec<i64>>) -> Result<[usize; 4], GpuError> {
    match p.as_deref() {
        None => Ok([0; 4]),
        Some([a, b, c, d]) => Ok([dim_of(*a)?, dim_of(*b)?, dim_of(*c)?, dim_of(*d)?]),
        Some(other) => Err(model(format!("pads {other:?}"))),
    }
}

/// A 2D attribute (kernel_shape, strides) as two positive sizes.
fn pair(p: Option<Vec<i64>>, default: [usize; 2], what: &str) -> Result<[usize; 2], GpuError> {
    match p.as_deref() {
        None => Ok(default),
        Some([a, b]) if *a > 0 && *b > 0 => Ok([dim_of(*a)?, dim_of(*b)?]),
        Some(other) => Err(model(format!("{what} {other:?}"))),
    }
}

/// A Conv with its weights laid out for the kernel it uses.
struct ConvW {
    w: Buf,
    b: Buf,
    cout: usize,
    cin_g: usize,
    group: usize,
    kh: usize,
    kw: usize,
    stride: [usize; 2],
    pads: [usize; 4],
    same_upper: bool,
    fast: bool,
}

/// What a fused node does.
enum Fused {
    /// The conv, its activation, a tensor added after it (a residual), and a [n, cin] scale its
    /// input is multiplied by first (squeeze-excitation).
    Conv {
        conv: ConvW,
        act: u32,
        residual: Option<String>,
        scale: Option<String>,
    },
    /// MatMul with a constant [K, N] weight, its bias, its activation.
    Linear {
        w: Buf,
        b: Buf,
        k: usize,
        n: usize,
        act: u32,
    },
    Affine {
        st: Buf,
    },
    /// The exporter's LayerNorm chain over the last axis.
    LayerNorm {
        gb: Buf,
        len: usize,
        eps: f32,
    },
}

pub(crate) struct NdGraph {
    nodes: Vec<Node>,
    /// Float initializers unfused nodes read, on the GPU, with their shapes.
    weights: HashMap<String, Tensor>,
    ints: HashMap<String, Vec<i64>>,
    fused: HashMap<usize, Fused>,
    skip: Vec<bool>,
    /// The last node reading each value.
    last_use: HashMap<String, usize>,
    input: String,
    /// The input of the graph's final Softmax: the logits.
    output: String,
}

/// The activation code (as `activate` in the shaders) of a node that can follow a Conv.
fn conv_act(op: &str) -> Option<u32> {
    match op {
        "Relu" => Some(1),
        "HardSwish" => Some(5),
        _ => None,
    }
}

impl NdGraph {
    /// The graph up to its final Softmax over the last axis, whose input it returns.
    pub(crate) fn new(gpu: &Gpu, onnx: &Onnx) -> Result<Self, GpuError> {
        // Identity nodes: every reader reads their input instead.
        let mut alias: HashMap<String, String> = HashMap::new();
        for n in onnx.nodes.iter().filter(|n| n.op == "Identity") {
            let src = alias.get(n.input(0)).cloned().unwrap_or_else(|| n.input(0).to_string());
            alias.insert(n.output().to_string(), src);
        }
        let resolve = |s: &str| alias.get(s).cloned().unwrap_or_else(|| s.to_string());
        let mut nodes: Vec<Node> = onnx
            .nodes
            .iter()
            .filter(|n| n.op != "Identity")
            .map(|n| {
                let mut n = n.clone();
                n.inputs = n.inputs.iter().map(|i| resolve(i)).collect();
                n
            })
            .collect();
        let graph_out = resolve(&onnx.output);
        let last = nodes.iter().rposition(|n| n.output() == graph_out).ok_or_else(|| model("the graph output is computed by no node"))?;
        let softmax = nodes.get(last).filter(|n| n.op == "Softmax").ok_or_else(|| model("the recognition model does not end in a Softmax"))?;
        if softmax.int("axis").is_some_and(|a| a != -1 && a != 2) {
            return Err(model("the final Softmax is not over the classes"));
        }
        let output = softmax.input(0).to_string();
        nodes.truncate(last);

        let scalar = |name: &str| onnx.weights.get(name).filter(|w| w.data.len() == 1).and_then(|w| w.data.first().copied());
        let count_readers = |nodes: &[Node], skip: &[bool]| {
            let mut c: HashMap<String, usize> = HashMap::new();
            for (k, n) in nodes.iter().enumerate() {
                if !skip.get(k).copied().unwrap_or(false) {
                    for i in &n.inputs {
                        *c.entry(i.clone()).or_default() += 1;
                    }
                }
            }
            c
        };
        let (mut fused, mut skip) = (HashMap::new(), vec![false; nodes.len()]);

        // Squeeze-excitation: Mul(x, s) with s from a HardSigmoid, read only by a dense Conv,
        // becomes that conv's input scale.
        let first = count_readers(&nodes, &skip);
        let readers = |name: &str| first.get(name).copied().unwrap_or(0) + usize::from(name == output);
        let mut in_scale: HashMap<usize, String> = HashMap::new();
        let mut se: Vec<(usize, usize, String, String)> = Vec::new();
        for (mi, m) in nodes.iter().enumerate() {
            if m.op != "Mul" || readers(m.output()) != 1 {
                continue;
            }
            let Some(ci) = nodes.iter().position(|c| c.op == "Conv" && c.input(0) == m.output() && c.int("group").unwrap_or(1) == 1) else {
                continue;
            };
            let gate = |name: &str| nodes.iter().any(|g| g.op == "HardSigmoid" && g.output() == name);
            let (x, sc) = if gate(m.input(1)) {
                (m.input(0), m.input(1))
            } else if gate(m.input(0)) {
                (m.input(1), m.input(0))
            } else {
                continue;
            };
            se.push((mi, ci, x.to_string(), sc.to_string()));
        }
        for (mi, ci, x, sc) in se {
            in_scale.insert(ci, sc);
            if let Some(s) = skip.get_mut(mi) {
                *s = true;
            }
            if let Some(i) = nodes.get_mut(ci).and_then(|c| c.inputs.first_mut()) {
                *i = x;
            }
        }

        let counts = count_readers(&nodes, &skip);
        let readers = |name: &str| counts.get(name).copied().unwrap_or(0) + usize::from(name == output);
        let is_const = |name: &str| onnx.weights.contains_key(name) || onnx.ints.contains_key(name);
        // (node, new output): a fused node writes its last swallowed node's output.
        let mut renames: Vec<(usize, String)> = Vec::new();
        let mut i = 0;
        while let Some(n) = nodes.get(i) {
            if skip.get(i).copied().unwrap_or(false) {
                i += 1;
                continue;
            }
            // LayerNorm: ReduceMean, Sub, Pow 2, ReduceMean, Add eps, Sqrt, Div, Mul gamma, Add beta.
            if n.op == "ReduceMean"
                && let Some([mean, sub, pow, var, add, sqrt, div, mul, beta]) = nodes.get(i..i + 9)
                && mean.ints("axes").as_deref() == Some(&[-1][..])
                && sub.op == "Sub"
                && sub.input(0) == mean.input(0)
                && sub.input(1) == mean.output()
                && pow.op == "Pow"
                && pow.input(0) == sub.output()
                && scalar(pow.input(1)) == Some(2.0)
                && var.op == "ReduceMean"
                && var.input(0) == pow.output()
                && var.ints("axes").as_deref() == Some(&[-1][..])
                && add.op == "Add"
                && add.input(0) == var.output()
                && scalar(add.input(1)).is_some()
                && sqrt.op == "Sqrt"
                && sqrt.input(0) == add.output()
                && div.op == "Div"
                && div.input(0) == sub.output()
                && div.input(1) == sqrt.output()
                && mul.op == "Mul"
                && mul.input(0) == div.output()
                && onnx.weights.contains_key(mul.input(1))
                && beta.op == "Add"
                && beta.input(0) == mul.output()
                && onnx.weights.contains_key(beta.input(1))
                && readers(mean.output()) == 1
                && readers(sub.output()) == 2
                && [pow.output(), var.output(), add.output(), sqrt.output(), div.output(), mul.output()].iter().all(|o| readers(o) == 1)
            {
                let (g, b) = (onnx.weight(mul.input(1))?, onnx.weight(beta.input(1))?);
                if g.data.len() != b.data.len() || g.data.is_empty() {
                    return Err(model("LayerNorm gamma and beta differ in length"));
                }
                let mut gb = g.data.clone();
                gb.extend_from_slice(&b.data);
                let eps = scalar(add.input(1)).unwrap_or(1e-5);
                fused.insert(i, Fused::LayerNorm { gb: gpu.upload(&gb)?, len: g.data.len(), eps });
                for s in skip.iter_mut().take(i + 9).skip(i + 1) {
                    *s = true;
                }
                renames.push((i, beta.output().to_string()));
                i += 9;
                continue;
            }
            let mut j = i + 1;
            let mut out = n.output().to_string();
            // A following Add of a constant bias of `want` values, read only here.
            let takes_bias = |next: &Node, out: &str, want: usize| -> Option<String> {
                if next.op != "Add" || readers(out) != 1 {
                    return None;
                }
                let other = if next.input(0) == out {
                    next.input(1)
                } else if next.input(1) == out {
                    next.input(0)
                } else {
                    return None;
                };
                let w = onnx.weights.get(other)?;
                (w.data.len() == want && w.dims.iter().filter(|&&d| d != 1).count() <= 1).then(|| other.to_string())
            };
            match n.op.as_str() {
                "Conv" => {
                    let w = onnx.weight(n.input(1))?;
                    let [cout, cin_g, kh, kw] = <[usize; 4]>::try_from(w.dims.clone()).map_err(|_| model(format!("conv weight {:?}", w.dims)))?;
                    if cout == 0 || cin_g == 0 || kh == 0 || kw == 0 {
                        return Err(model(format!("conv weight {:?}", w.dims)));
                    }
                    let mut b = match n.inputs.get(2).filter(|s| !s.is_empty()) {
                        Some(s) => onnx.weight(s)?.data.clone(),
                        None => vec![0.0; cout],
                    };
                    if b.len() != cout {
                        return Err(model(format!("conv bias of {} for {cout} channels", b.len())));
                    }
                    if let Some(next) = nodes.get(j)
                        && let Some(name) = takes_bias(next, &out, cout)
                    {
                        for (x, y) in b.iter_mut().zip(&onnx.weight(&name)?.data) {
                            *x += y;
                        }
                        out = next.output().to_string();
                        j += 1;
                    }
                    let mut act = 0;
                    if let Some(next) = nodes.get(j)
                        && let Some(a) = conv_act(&next.op)
                        && next.input(0) == out
                        && readers(&out) == 1
                    {
                        act = a;
                        out = next.output().to_string();
                        j += 1;
                    }
                    // GELU as the exporter writes it: x * (erf(x / sqrt 2) + 1) * 0.5.
                    if act == 0
                        && let Some([div, erf, add, mul, half]) = nodes.get(j..j + 5)
                        && div.op == "Div"
                        && div.input(0) == out
                        && scalar(div.input(1)).is_some_and(|c| (c - std::f32::consts::SQRT_2).abs() < 1e-4)
                        && erf.op == "Erf"
                        && erf.input(0) == div.output()
                        && add.op == "Add"
                        && add.inputs.iter().any(|i| i == erf.output())
                        && add.inputs.iter().any(|i| scalar(i).is_some_and(|c| (c - 1.0).abs() < 1e-6))
                        && mul.op == "Mul"
                        && mul.inputs.contains(&out)
                        && mul.inputs.iter().any(|i| i == add.output())
                        && half.op == "Mul"
                        && half.inputs.iter().any(|i| i == mul.output())
                        && half.inputs.iter().any(|i| scalar(i).is_some_and(|c| (c - 0.5).abs() < 1e-6))
                        && readers(&out) == 2
                        && [div.output(), erf.output(), add.output(), mul.output()].iter().all(|o| readers(o) == 1)
                    {
                        act = 3;
                        out = half.output().to_string();
                        j += 5;
                    }
                    let group = usize::try_from(n.int("group").unwrap_or(1)).ok().filter(|&g| g > 0).ok_or_else(|| model("conv group"))?;
                    // A residual Add of another computed tensor (dense convolutions only).
                    let mut residual = None;
                    if group == 1
                        && let Some(next) = nodes.get(j)
                        && next.op == "Add"
                        && readers(&out) == 1
                        && next.inputs.contains(&out)
                        && let Some(other) = next.inputs.iter().find(|i| **i != out)
                        && !is_const(other)
                    {
                        residual = Some(other.clone());
                        out = next.output().to_string();
                        j += 1;
                    }
                    if n.ints("dilations").is_some_and(|d| d.iter().any(|&x| x != 1)) {
                        return Err(model("dilated conv"));
                    }
                    let stride = pair(n.ints("strides"), [1, 1], "conv strides")?;
                    let auto = n.string("auto_pad").unwrap_or_default();
                    if !matches!(auto.as_str(), "" | "NOTSET" | "SAME_UPPER") {
                        return Err(model(format!("conv auto_pad {auto}")));
                    }
                    let pads = pads4(n.ints("pads"))?;
                    let fast = group == 1 && cin_g.is_multiple_of(16);
                    let data = if group == 1 {
                        // [co][ci][ky][kx] -> [k][co], k = (ky, kx, ci) when fast, else (ci, ky, kx).
                        let mut t = vec![0f32; w.data.len()];
                        for co in 0..cout {
                            for ci in 0..cin_g {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let k = if fast { (ky * kw + kx) * cin_g + ci } else { (ci * kh + ky) * kw + kx };
                                        if let (Some(dst), Some(&v)) = (t.get_mut(k * cout + co), w.data.get(((co * cin_g + ci) * kh + ky) * kw + kx))
                                        {
                                            *dst = v;
                                        }
                                    }
                                }
                            }
                        }
                        t
                    } else {
                        w.data.clone()
                    };
                    let conv = ConvW {
                        w: gpu.upload(&data)?,
                        b: gpu.upload(&b)?,
                        cout,
                        cin_g,
                        group,
                        kh,
                        kw,
                        stride,
                        pads,
                        same_upper: auto == "SAME_UPPER",
                        fast,
                    };
                    fused.insert(i, Fused::Conv { conv, act, residual, scale: in_scale.get(&i).cloned() });
                }
                "MatMul" if onnx.weights.get(n.input(1)).is_some_and(|w| w.dims.len() == 2) => {
                    let w = onnx.weight(n.input(1))?;
                    let (k, nn) = (dim(&w.dims, 0)?, dim(&w.dims, 1)?);
                    let mut b = vec![0.0; nn];
                    if let Some(next) = nodes.get(j)
                        && let Some(name) = takes_bias(next, &out, nn)
                    {
                        b.copy_from_slice(onnx.weight(&name)?.data.get(..nn).ok_or_else(|| model("MatMul bias"))?);
                        out = next.output().to_string();
                        j += 1;
                    }
                    // SiLU: x * sigmoid(x).
                    let mut act = 0;
                    if let Some([sig, mul]) = nodes.get(j..j + 2)
                        && sig.op == "Sigmoid"
                        && sig.input(0) == out
                        && mul.op == "Mul"
                        && mul.inputs.contains(&out)
                        && mul.inputs.iter().any(|i| i == sig.output())
                        && readers(&out) == 2
                        && readers(sig.output()) == 1
                    {
                        act = 4;
                        out = mul.output().to_string();
                        j += 2;
                    }
                    fused.insert(i, Fused::Linear { w: gpu.upload(&w.data)?, b: gpu.upload(&b)?, k, n: nn, act });
                }
                "BatchNormalization" => {
                    let g = |k: usize| onnx.weight(n.input(k)).map(|w| w.data.clone());
                    let (scale, shift, mean, var) = (g(1)?, g(2)?, g(3)?, g(4)?);
                    let eps = n.float("epsilon").unwrap_or(1e-5);
                    let c = scale.len();
                    if [shift.len(), mean.len(), var.len()].iter().any(|&l| l != c) {
                        return Err(model("BatchNormalization parameters differ in length"));
                    }
                    let mut st = vec![0f32; 2 * c];
                    for ch in 0..c {
                        let (sc, sh, me, va) = (scale.get(ch), shift.get(ch), mean.get(ch), var.get(ch));
                        if let (Some(&sc), Some(&sh), Some(&me), Some(&va)) = (sc, sh, me, va) {
                            let s = sc / (va + eps).sqrt();
                            if let Some(x) = st.get_mut(ch) {
                                *x = s;
                            }
                            if let Some(x) = st.get_mut(c + ch) {
                                *x = sh - me * s;
                            }
                        }
                    }
                    fused.insert(i, Fused::Affine { st: gpu.upload(&st)? });
                }
                _ => {}
            }
            for s in skip.iter_mut().take(j).skip(i + 1) {
                *s = true;
            }
            if j > i + 1 {
                renames.push((i, out.clone()));
            }
            i = j;
        }
        for (i, out) in renames {
            if let Some(o) = nodes.get_mut(i).and_then(|n| n.outputs.first_mut()) {
                *o = out;
            }
        }
        // Only the float constants unfused nodes read go to the GPU; fused nodes carry their own.
        let mut used: HashSet<&str> = HashSet::new();
        for (k, n) in nodes.iter().enumerate() {
            if skip.get(k).copied().unwrap_or(false) || fused.contains_key(&k) {
                continue;
            }
            used.extend(n.inputs.iter().map(String::as_str));
        }
        let mut weights = HashMap::new();
        for (name, w) in onnx.weights.iter().filter(|(name, _)| used.contains(name.as_str())) {
            weights.insert(name.clone(), Tensor { buf: gpu.upload(&w.data)?, shape: w.dims.clone() });
        }
        let mut graph = Self { nodes, weights, ints: onnx.ints.clone(), fused, skip, last_use: HashMap::new(), input: onnx.input.clone(), output };
        let mut last_use = HashMap::new();
        for (k, n) in graph.nodes.iter().enumerate() {
            if !graph.skip.get(k).copied().unwrap_or(false) {
                for name in graph.reads(k, n) {
                    last_use.insert(name.to_string(), k);
                }
            }
        }
        graph.last_use = last_use;
        Ok(graph)
    }

    /// The values node `idx` reads: its inputs, and a fused conv's residual and input scale.
    fn reads<'a>(&'a self, idx: usize, n: &'a Node) -> impl Iterator<Item = &'a str> + 'a {
        let extra = match self.fused.get(&idx) {
            Some(Fused::Conv { residual, scale, .. }) => [residual.as_deref(), scale.as_deref()],
            _ => [None, None],
        };
        n.inputs.iter().map(String::as_str).chain(extra.into_iter().flatten())
    }

    /// The length of the logits' last axis, when a linear layer with a constant weight writes
    /// them (the CTC head of the PP-OCR models).
    pub(crate) fn classes(&self) -> Option<usize> {
        let at = self.nodes.iter().enumerate().rposition(|(k, n)| !self.skip.get(k).copied().unwrap_or(false) && n.output() == self.output)?;
        match self.fused.get(&at) {
            Some(Fused::Linear { n, .. }) => Some(*n),
            _ => None,
        }
    }

    fn get(&self, values: &HashMap<String, Value>, name: &str) -> Result<Value, GpuError> {
        if let Some(v) = values.get(name) {
            return Ok(v.clone());
        }
        if let Some(t) = self.weights.get(name) {
            return Ok(Value::F(t.clone()));
        }
        if let Some(i) = self.ints.get(name) {
            return Ok(Value::I(i.clone(), if i.len() == 1 { vec![1] } else { vec![i.len()] }));
        }
        Err(model(format!("missing value {name}")))
    }

    /// Record the graph on `x`; returns the logits (the input of the final Softmax). A value's
    /// buffer goes back to the pool once its last reader is recorded (wgpu orders a later write to
    /// it after those reads), unless another value still shares it (a Reshape's output does), so a
    /// run holds only the values still to be read.
    pub(crate) fn run(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, x: Tensor) -> Result<Tensor, GpuError> {
        let mut values: HashMap<String, Value> = HashMap::new();
        values.insert(self.input.clone(), Value::F(x));
        for (idx, n) in self.nodes.iter().enumerate() {
            if self.skip.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let out = self.node(gpu, enc, idx, n, &values)?;
            values.insert(n.output().to_string(), out);
            for name in self.reads(idx, n) {
                if self.last_use.get(name) == Some(&idx)
                    && name != self.output
                    && let Some(Value::F(t)) = values.remove(name)
                    && std::sync::Arc::strong_count(&t.buf) == 1
                {
                    gpu.recycle([t.buf]);
                }
            }
        }
        match values.remove(&self.output) {
            Some(Value::F(t)) => Ok(t),
            _ => Err(model(format!("output {} not computed", self.output))),
        }
    }

    fn node(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, idx: usize, n: &Node, values: &HashMap<String, Value>) -> Result<Value, GpuError> {
        let new = |shape: &[usize]| -> Result<Tensor, GpuError> {
            if shape.len() > RANK {
                return Err(model(format!("rank {} above {RANK}", shape.len())));
            }
            Ok(Tensor { buf: gpu.storage_rounded(count(shape)?)?, shape: shape.to_vec() })
        };
        let arg = |k: usize| self.get(values, n.input(k));
        let float = |k: usize| -> Result<Tensor, GpuError> {
            match arg(k)? {
                Value::F(t) => Ok(t),
                Value::I(..) => Err(model(format!("{}: input {k} is not float", n.op))),
            }
        };
        let ints = |k: usize| -> Result<Vec<i64>, GpuError> {
            match arg(k)? {
                Value::I(v, _) => Ok(v),
                Value::F(_) => Err(model(format!("{}: input {k} is not an integer constant", n.op))),
            }
        };
        let has = |k: usize| n.inputs.get(k).is_some_and(|s| !s.is_empty());
        Ok(match (n.op.as_str(), self.fused.get(&idx)) {
            ("Conv", Some(Fused::Conv { conv, act, residual, scale })) => {
                let tensor = |name: &Option<String>| -> Result<Option<Tensor>, GpuError> {
                    match name {
                        Some(name) => match self.get(values, name)? {
                            Value::F(t) => Ok(Some(t)),
                            Value::I(..) => Err(model("integer input to a fused conv")),
                        },
                        None => Ok(None),
                    }
                };
                let (r, s) = (tensor(residual)?, tensor(scale)?);
                Value::F(self.conv(gpu, enc, &float(0)?, conv, *act, r.as_ref(), s.as_ref())?)
            }
            ("GlobalAveragePool", _) => {
                let a = float(0)?;
                if a.shape.len() < 3 {
                    return Err(model(format!("GlobalAveragePool on {:?}", a.shape)));
                }
                let inner = count(a.shape.get(2..).unwrap_or_default())?;
                let outer = count(a.shape.get(..2).unwrap_or_default())?;
                let shape: Vec<usize> = a.shape.iter().enumerate().map(|(d, &s)| if d >= 2 { 1 } else { s }).collect();
                let y = new(&shape)?;
                self.mean_rows(gpu, enc, &a.buf, &y.buf, outer, inner)?;
                Value::F(y)
            }
            ("ReduceMean", Some(Fused::LayerNorm { gb, len, eps })) => {
                let a = float(0)?;
                if a.shape.last() != Some(len) {
                    return Err(model(format!("LayerNorm over {:?}, expects {len}", a.shape)));
                }
                let rows = count(&a.shape)? / len.max(&1);
                let y = new(&a.shape)?;
                let p = gpu.uniform(&[to_u32(rows)?, to_u32(*len)?, eps.to_bits(), 0])?;
                gpu.elementwise(enc, "nd_layer_norm", &[&a.buf, gb, &y.buf, &p], rows)?;
                Value::F(y)
            }
            ("MatMul", Some(Fused::Linear { w, b, k, n: nn, act })) => {
                let a = float(0)?;
                if a.shape.last() != Some(k) {
                    return Err(model(format!("MatMul: {:?} by [{k}, {nn}]", a.shape)));
                }
                let m = count(a.shape.get(..a.shape.len().saturating_sub(1)).unwrap_or_default())?;
                let mut shape = a.shape.clone();
                if let Some(l) = shape.last_mut() {
                    *l = *nn;
                }
                let y = new(&shape)?;
                self.gemm(gpu, enc, &a.buf, w, b, &y.buf, [m, *nn, *k, 1], [0, 0, 0], *act)?;
                Value::F(y)
            }
            ("MatMul", None) => {
                let (a, bm) = (float(0)?, float(1)?);
                let r = a.shape.len();
                let lead = |s: &[usize]| s.get(..r.saturating_sub(2)).map(<[usize]>::to_vec);
                if r < 2 || bm.shape.len() != r || lead(&a.shape) != lead(&bm.shape) || dim(&a.shape, r - 1)? != dim(&bm.shape, r - 2)? {
                    return Err(model(format!("MatMul shapes {:?} x {:?}", a.shape, bm.shape)));
                }
                let (m, k, nn) = (dim(&a.shape, r - 2)?, dim(&a.shape, r - 1)?, dim(&bm.shape, r - 1)?);
                let batch = count(a.shape.get(..r - 2).unwrap_or_default())?;
                let mut shape = a.shape.clone();
                if let Some(l) = shape.last_mut() {
                    *l = nn;
                }
                let y = new(&shape)?;
                let zeros = gpu.upload(&vec![0f32; nn])?;
                self.gemm(gpu, enc, &a.buf, &bm.buf, &zeros, &y.buf, [m, nn, k, batch], [count(&[m, k])?, count(&[k, nn])?, count(&[m, nn])?], 0)?;
                Value::F(y)
            }
            ("BatchNormalization", Some(Fused::Affine { st })) => {
                let a = float(0)?;
                let c = a.shape.get(1).copied().unwrap_or(1);
                let inner = count(a.shape.get(2..).unwrap_or_default())?;
                let y = new(&a.shape)?;
                let total = count(&a.shape)?;
                let p = gpu.uniform(&params(&[total, c, inner.max(1), 0])?)?;
                gpu.elementwise(enc, "nd_affine", &[&a.buf, st, &y.buf, &p], total)?;
                Value::F(y)
            }
            ("Add" | "Sub" | "Mul" | "Div" | "Pow", _) => match (arg(0)?, arg(1)?) {
                (Value::F(a), Value::F(b)) => Value::F(self.binary(gpu, enc, &n.op, &a, &b)?),
                (Value::I(a, sa), Value::I(b, _)) => {
                    let f = |x: i64, y: i64| -> Result<i64, GpuError> {
                        let r = match n.op.as_str() {
                            "Add" => x.checked_add(y),
                            "Sub" => x.checked_sub(y),
                            "Mul" => x.checked_mul(y),
                            "Div" => x.checked_div(y),
                            _ => None,
                        };
                        r.ok_or_else(|| model(format!("integer {} of {x} and {y}", n.op)))
                    };
                    let v: Vec<i64> = match b.as_slice() {
                        [y] => a.iter().map(|&x| f(x, *y)).collect::<Result<_, _>>()?,
                        _ if a.len() == b.len() => a.iter().zip(&b).map(|(&x, &y)| f(x, y)).collect::<Result<_, _>>()?,
                        _ => return Err(model(format!("integer {} of {} and {} values", n.op, a.len(), b.len()))),
                    };
                    Value::I(v, sa)
                }
                _ => return Err(model(format!("{}: mixed integer and float inputs", n.op))),
            },
            ("Relu" | "Sigmoid" | "HardSigmoid" | "Erf" | "Sqrt" | "HardSwish", _) => {
                let a = float(0)?;
                let op = match n.op.as_str() {
                    "Relu" => 1,
                    "Sigmoid" => 2,
                    "HardSigmoid" => 3,
                    "Erf" => 4,
                    "Sqrt" => 5,
                    _ => 6,
                };
                let (alpha, beta) = (n.float("alpha").unwrap_or(0.2), n.float("beta").unwrap_or(0.5));
                let y = new(&a.shape)?;
                let total = count(&a.shape)?;
                let p = gpu.uniform(&[to_u32(total)?, op, alpha.to_bits(), beta.to_bits()])?;
                gpu.elementwise(enc, "nd_unary", &[&a.buf, &y.buf, &p], total)?;
                Value::F(y)
            }
            ("ReduceMean", _) => {
                let a = float(0)?;
                let r = a.shape.len();
                let all: Vec<i64> = (0..r as i64).collect();
                let mut axes: Vec<usize> = n.ints("axes").unwrap_or(all).iter().map(|&x| axis(x, r)).collect::<Result<_, _>>()?;
                axes.sort_unstable();
                axes.dedup();
                if axes.iter().enumerate().any(|(k, &x)| x != r - axes.len() + k) {
                    return Err(model(format!("ReduceMean over non-trailing axes {axes:?}")));
                }
                let inner = count(&axes.iter().map(|&x| dim(&a.shape, x)).collect::<Result<Vec<_>, _>>()?)?;
                let outer = count(&a.shape)? / inner.max(1);
                let keepdims = n.int("keepdims").unwrap_or(1) == 1;
                let shape: Vec<usize> =
                    a.shape.iter().enumerate().filter_map(|(d, &s)| if axes.contains(&d) { keepdims.then_some(1) } else { Some(s) }).collect();
                let y = new(&shape)?;
                self.mean_rows(gpu, enc, &a.buf, &y.buf, outer, inner)?;
                Value::F(y)
            }
            ("Softmax", _) => {
                let a = float(0)?;
                let ax = axis(n.int("axis").unwrap_or(-1), a.shape.len())?;
                let len = count(a.shape.get(ax..).unwrap_or_default())?;
                let rows = count(&a.shape)? / len.max(1);
                let y = new(&a.shape)?;
                let p = gpu.uniform(&params(&[rows, len, 0, 0])?)?;
                gpu.elementwise(enc, "nd_softmax", &[&a.buf, &y.buf, &p], rows)?;
                Value::F(y)
            }
            ("Transpose", _) => {
                let a = float(0)?;
                let r = a.shape.len();
                let perm: Vec<usize> = match n.ints("perm") {
                    Some(p) => p.iter().map(|&x| axis(x, r)).collect::<Result<_, _>>()?,
                    None => (0..r).rev().collect(),
                };
                let mut seen = perm.clone();
                seen.sort_unstable();
                if seen != (0..r).collect::<Vec<_>>() {
                    return Err(model(format!("Transpose perm {perm:?}")));
                }
                let shape: Vec<usize> = perm.iter().map(|&p| dim(&a.shape, p)).collect::<Result<_, _>>()?;
                let is = strides(&a.shape)?;
                let y = new(&shape)?;
                let src: Vec<usize> = perm.iter().map(|&p| is.get(p).copied().unwrap_or(0)).collect();
                self.copy(gpu, enc, &a.buf, &y.buf, &shape, &src, &strides(&shape)?, 0, 0)?;
                Value::F(y)
            }
            ("Slice", _) => {
                let (starts, ends) = (ints(1)?, ints(2)?);
                let data = arg(0)?;
                let shape = match &data {
                    Value::F(t) => t.shape.clone(),
                    Value::I(_, s) => s.clone(),
                };
                let r = shape.len();
                let axes = if has(3) { ints(3)? } else { (0..starts.len() as i64).collect() };
                if has(4) && ints(4)?.iter().any(|&s| s != 1) {
                    return Err(model("Slice with steps"));
                }
                if starts.len() != ends.len() || starts.len() != axes.len() {
                    return Err(model("Slice starts, ends and axes differ in length"));
                }
                let mut lo = vec![0usize; r];
                let mut out = shape.clone();
                for ((&st, &en), &ax) in starts.iter().zip(&ends).zip(&axes) {
                    let ax = axis(ax, r)?;
                    let d = i64::try_from(dim(&shape, ax)?).map_err(|_| model("Slice size"))?;
                    let clampv = |v: i64| (if v < 0 { v.saturating_add(d) } else { v }).clamp(0, d) as usize;
                    let (s0, e0) = (clampv(st), clampv(en));
                    if let (Some(l), Some(o)) = (lo.get_mut(ax), out.get_mut(ax)) {
                        *l = s0;
                        *o = e0.saturating_sub(s0);
                    }
                }
                match data {
                    Value::I(v, _) => {
                        // A 1-D integer slice (shape arithmetic).
                        let (l, o) = (lo.first().copied().unwrap_or(0), out.first().copied().unwrap_or(0));
                        Value::I(v.get(l..l.saturating_add(o)).map(<[i64]>::to_vec).unwrap_or_default(), out)
                    }
                    Value::F(a) => {
                        let is = strides(&a.shape)?;
                        let base = lo.iter().zip(&is).try_fold(0usize, |acc, (l, s)| l.checked_mul(*s).and_then(|v| acc.checked_add(v)));
                        let base = base.ok_or_else(|| GpuError::TooLarge("Slice".into()))?;
                        let y = new(&out)?;
                        self.copy(gpu, enc, &a.buf, &y.buf, &out, &is, &strides(&out)?, base, 0)?;
                        Value::F(y)
                    }
                }
            }
            ("Concat", _) => {
                let parts: Vec<Value> = (0..n.inputs.len()).map(arg).collect::<Result<_, _>>()?;
                if parts.iter().all(|p| matches!(p, Value::I(..))) {
                    let v: Vec<i64> = parts.iter().flat_map(|p| if let Value::I(v, _) = p { v.clone() } else { Vec::new() }).collect();
                    let len = v.len();
                    Value::I(v, vec![len])
                } else {
                    let ts: Vec<Tensor> = parts
                        .into_iter()
                        .map(|p| if let Value::F(t) = p { Ok(t) } else { Err(model("Concat of integer and float values")) })
                        .collect::<Result<_, _>>()?;
                    let first = ts.first().ok_or_else(|| model("Concat of nothing"))?;
                    let r = first.shape.len();
                    let ax = axis(n.int("axis").unwrap_or(0), r)?;
                    let mut shape = first.shape.clone();
                    for t in ts.iter().skip(1) {
                        if t.shape.len() != r || t.shape.iter().zip(&first.shape).enumerate().any(|(d, (a, b))| d != ax && a != b) {
                            return Err(model(format!("Concat of {:?} and {:?}", first.shape, t.shape)));
                        }
                    }
                    let total = ts.iter().try_fold(0usize, |acc, t| t.shape.get(ax).and_then(|d| acc.checked_add(*d)));
                    if let Some(s) = shape.get_mut(ax) {
                        *s = total.ok_or_else(|| GpuError::TooLarge("Concat".into()))?;
                    }
                    let y = new(&shape)?;
                    let os = strides(&shape)?;
                    let step = os.get(ax).copied().unwrap_or(0);
                    let mut off = 0usize;
                    for t in &ts {
                        let obase = off.checked_mul(step).ok_or_else(|| GpuError::TooLarge("Concat".into()))?;
                        self.copy(gpu, enc, &t.buf, &y.buf, &t.shape, &strides(&t.shape)?, &os, 0, obase)?;
                        off = off.saturating_add(dim(&t.shape, ax)?);
                    }
                    Value::F(y)
                }
            }
            ("MaxPool" | "AveragePool", _) => Value::F(self.pool(gpu, enc, n, &float(0)?)?),
            ("Reshape", _) => {
                let a = float(0)?;
                let want = ints(1)?;
                let total = count(&a.shape)?;
                let mut shape: Vec<usize> = Vec::with_capacity(want.len());
                for (d, &s) in want.iter().enumerate() {
                    shape.push(match s {
                        0 => a.shape.get(d).copied().unwrap_or(1),
                        -1 => 1,
                        s => dim_of(s)?,
                    });
                }
                match want.iter().filter(|&&s| s == -1).count() {
                    0 => {}
                    1 => {
                        let known = count(&shape)?;
                        if let Some(p) = want.iter().position(|&s| s == -1).and_then(|p| shape.get_mut(p)) {
                            *p = total.checked_div(known).unwrap_or(0);
                        }
                    }
                    _ => return Err(model(format!("Reshape to {want:?}"))),
                }
                if count(&shape)? != total || shape.len() > RANK {
                    return Err(model(format!("Reshape {:?} to {want:?}", a.shape)));
                }
                Value::F(Tensor { buf: a.buf, shape })
            }
            ("Squeeze" | "Unsqueeze", _) => {
                let axes = match n.ints("axes") {
                    Some(a) => a,
                    None if has(1) => ints(1)?,
                    None => Vec::new(),
                };
                match arg(0)? {
                    Value::F(a) => {
                        let shape = if n.op == "Squeeze" {
                            let ax: Vec<usize> = axes.iter().map(|&x| axis(x, a.shape.len())).collect::<Result<_, _>>()?;
                            if ax.is_empty() {
                                a.shape.iter().copied().filter(|&d| d != 1).collect()
                            } else {
                                a.shape.iter().enumerate().filter(|(d, _)| !ax.contains(d)).map(|(_, &s)| s).collect()
                            }
                        } else {
                            let r = a.shape.len() + axes.len();
                            let ax: Vec<usize> = axes.iter().map(|&x| axis(x, r)).collect::<Result<_, _>>()?;
                            let mut it = a.shape.iter();
                            (0..r).map(|d| if ax.contains(&d) { 1 } else { it.next().copied().unwrap_or(1) }).collect::<Vec<_>>()
                        };
                        if shape.len() > RANK {
                            return Err(model(format!("rank {} above {RANK}", shape.len())));
                        }
                        Value::F(Tensor { buf: a.buf, shape })
                    }
                    Value::I(v, s) => {
                        let shape = if n.op == "Squeeze" { s.into_iter().filter(|&d| d != 1).collect() } else { vec![v.len()] };
                        Value::I(v, shape)
                    }
                }
            }
            ("Shape", _) => {
                let s = match arg(0)? {
                    Value::F(t) => t.shape,
                    Value::I(_, s) => s,
                };
                let v: Vec<i64> = s.iter().map(|&d| i64::try_from(d).map_err(|_| model("shape"))).collect::<Result<_, _>>()?;
                let len = v.len();
                Value::I(v, vec![len])
            }
            ("Gather", _) => {
                let (data, idx) = (ints(0)?, ints(1)?);
                let len = i64::try_from(data.len()).map_err(|_| model("Gather"))?;
                let v: Vec<i64> = idx
                    .iter()
                    .map(|&i| {
                        let at = if i < 0 { i.saturating_add(len) } else { i };
                        usize::try_from(at).ok().and_then(|at| data.get(at).copied()).ok_or_else(|| model(format!("Gather index {i} of {len}")))
                    })
                    .collect::<Result<_, _>>()?;
                let len = v.len();
                Value::I(v, if len == 1 { vec![] } else { vec![len] })
            }
            ("Cast", _) => arg(0)?,
            (other, _) => return Err(model(format!("unsupported op {other}"))),
        })
    }

    fn mean_rows(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, x: &Buf, y: &Buf, outer: usize, inner: usize) -> Result<(), GpuError> {
        let p = gpu.uniform(&params(&[outer, inner, 0, 0])?)?;
        if inner >= 512 {
            gpu.rows(enc, "nd_reduce_mean_wide", &[x, y, &p], outer)
        } else {
            gpu.elementwise(enc, "nd_reduce_mean", &[x, y, &p], outer)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        x: &Buf,
        w: &Buf,
        b: &Buf,
        y: &Buf,
        mkib: [usize; 4],
        strides: [usize; 3],
        act: u32,
    ) -> Result<(), GpuError> {
        let [m, k, i, batch] = mkib;
        let d = gpu.uniform(&params(&[m, k, i, act as usize, strides[0], strides[1], strides[2], 0])?)?;
        gpu.dispatch(enc, "nd_bgemm", &[x, w, b, y, &d], (to_u32(m.div_ceil(64))?, to_u32(k.div_ceil(64))?, to_u32(batch)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn copy(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        x: &Buf,
        y: &Buf,
        dims: &[usize],
        is: &[usize],
        os: &[usize],
        ibase: usize,
        obase: usize,
    ) -> Result<(), GpuError> {
        if dims.len() > RANK || is.len() != dims.len() || os.len() != dims.len() {
            return Err(model(format!("a copy of rank {}", dims.len())));
        }
        let total = count(dims)?;
        let mut p = vec![0usize; 22];
        let head = [total, dims.len(), ibase, obase];
        for (k, v) in head.iter().chain(pad6(dims).iter()).chain(pad6(is).iter()).chain(pad6(os).iter()).enumerate() {
            if let Some(x) = p.get_mut(k) {
                *x = *v;
            }
        }
        let pb = gpu.upload_u32(&params(&p)?)?;
        gpu.elementwise(enc, "nd_copy", &[x, y, &pb], total)
    }

    fn binary(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, op: &str, a: &Tensor, b: &Tensor) -> Result<Tensor, GpuError> {
        let r = a.shape.len().max(b.shape.len());
        if r > RANK {
            return Err(model(format!("rank {r} above {RANK}")));
        }
        let pad = |s: &[usize]| -> Vec<usize> { std::iter::repeat_n(1, r - s.len()).chain(s.iter().copied()).collect() };
        let (sa, sb) = (pad(&a.shape), pad(&b.shape));
        if sa.iter().zip(&sb).any(|(&x, &y)| x != y && x != 1 && y != 1) {
            return Err(model(format!("{op}: shapes {:?} and {:?} do not broadcast", a.shape, b.shape)));
        }
        let shape: Vec<usize> = sa.iter().zip(&sb).map(|(&x, &y)| if x == 1 { y } else { x }).collect();
        let bstr =
            |s: &[usize]| -> Result<Vec<usize>, GpuError> { Ok(strides(s)?.iter().zip(s).map(|(&st, &d)| if d == 1 { 0 } else { st }).collect()) };
        let total = count(&shape)?;
        let code = match op {
            "Add" => 0,
            "Sub" => 1,
            "Mul" => 2,
            "Div" => 3,
            _ => 4,
        };
        let mut p = vec![0usize; 22];
        let head = [total, r, code, 0];
        for (k, v) in head.iter().chain(pad6(&shape).iter()).chain(pad6(&bstr(&sa)?).iter()).chain(pad6(&bstr(&sb)?).iter()).enumerate() {
            if let Some(x) = p.get_mut(k) {
                *x = *v;
            }
        }
        let pb = gpu.upload_u32(&params(&p)?)?;
        let y = gpu.storage_rounded(total)?;
        gpu.elementwise(enc, "nd_binary", &[&a.buf, &b.buf, &y, &pb], total)?;
        Ok(Tensor { buf: y, shape })
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        a: &Tensor,
        c: &ConvW,
        act: u32,
        res: Option<&Tensor>,
        scale: Option<&Tensor>,
    ) -> Result<Tensor, GpuError> {
        let [n, cin, h, w] = <[usize; 4]>::try_from(a.shape.clone()).map_err(|_| model(format!("Conv input {:?}", a.shape)))?;
        let [sh, sw] = c.stride;
        let (pt, pl, pb, pr) = if c.same_upper {
            let ((t, b), (l, r)) = (same_upper(h, c.kh, sh), same_upper(w, c.kw, sw));
            (t, l, b, r)
        } else {
            let [t, l, b, r] = c.pads;
            (t, l, b, r)
        };
        let oh = out_len(h, pt.saturating_add(pb), c.kh, sh)?;
        let ow = out_len(w, pl.saturating_add(pr), c.kw, sw)?;
        let shape = [n, c.cout, oh, ow];
        if let Some(r) = res
            && r.shape != shape
        {
            return Err(model(format!("residual {:?} for a conv giving {shape:?}", r.shape)));
        }
        if let Some(s) = scale
            && count(&s.shape)? != count(&[n, cin])?
        {
            return Err(model(format!("input scale {:?} for {:?}", s.shape, a.shape)));
        }
        let y = gpu.storage_rounded(count(&shape)?)?;
        if c.group == 1 {
            if c.cin_g != cin {
                return Err(model(format!("Conv expects {} channels, has {cin}", c.cin_g)));
            }
            let flags = [usize::from(res.is_some()), usize::from(scale.is_some())];
            let p = gpu.uniform(&params(&[n, cin, h, w, c.cout, oh, ow, c.kh, c.kw, pt, sh, act as usize, pl, sw, flags[0], flags[1]])?)?;
            let (pipe, tile) = if c.fast && gpu.tuning.conv128 {
                ("conv128", 128)
            } else if c.fast {
                ("conv_fast", 64)
            } else {
                ("conv", 64)
            };
            let rb = res.map_or(&a.buf, |r| &r.buf);
            let sb = scale.map_or(&a.buf, |s| &s.buf);
            let pixels = count(&[n, oh, ow])?;
            gpu.dispatch(enc, pipe, &[&a.buf, &c.w, &c.b, &y, &p, rb, sb], (to_u32(pixels.div_ceil(tile))?, to_u32(c.cout.div_ceil(64))?, 1))?;
        } else if c.cin_g == 1 && c.group == cin && c.group == c.cout {
            if res.is_some() || scale.is_some() {
                return Err(model("residual or input scale on a depthwise conv"));
            }
            let p = gpu.uniform(&params(&[n, cin, h, w, oh, ow, c.kh, c.kw, pt, sh, act as usize, pl, sw])?)?;
            gpu.elementwise(enc, "depthwise", &[&a.buf, &c.w, &c.b, &y, &p], count(&shape)?)?;
        } else {
            return Err(model(format!("grouped conv with group {}", c.group)));
        }
        Ok(Tensor { buf: y, shape: shape.to_vec() })
    }

    fn pool(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, node: &Node, a: &Tensor) -> Result<Tensor, GpuError> {
        let [n, c, h, w] = <[usize; 4]>::try_from(a.shape.clone()).map_err(|_| model(format!("pool input {:?}", a.shape)))?;
        let [kh, kw] = pair(node.ints("kernel_shape"), [0, 0], "pool kernel_shape")?;
        if kh == 0 || kw == 0 {
            return Err(model("pool without kernel_shape"));
        }
        let [sh, sw] = pair(node.ints("strides"), [1, 1], "pool strides")?;
        if node.int("ceil_mode").unwrap_or(0) != 0 {
            return Err(model("pool with ceil_mode"));
        }
        if node.op == "AveragePool" && node.int("count_include_pad").unwrap_or(0) != 0 {
            return Err(model("AveragePool counting padding"));
        }
        let (pt, pl, pb, pr) = match node.string("auto_pad").as_deref() {
            Some("SAME_UPPER") => {
                let ((t, b), (l, r)) = (same_upper(h, kh, sh), same_upper(w, kw, sw));
                (t, l, b, r)
            }
            None | Some("" | "NOTSET") => {
                let [t, l, b, r] = pads4(node.ints("pads"))?;
                (t, l, b, r)
            }
            Some(other) => return Err(model(format!("pool auto_pad {other}"))),
        };
        let oh = out_len(h, pt.saturating_add(pb), kh, sh)?;
        let ow = out_len(w, pl.saturating_add(pr), kw, sw)?;
        let y = gpu.storage_rounded(count(&[n, c, oh, ow])?)?;
        let avg = usize::from(node.op == "AveragePool");
        let p = gpu.uniform(&params(&[count(&[n, c])?, h, w, oh, ow, kh, kw, sh, sw, pt, pl, avg])?)?;
        gpu.elementwise(enc, "nd_pool", &[&a.buf, &y, &p], count(&[n, c, oh, ow])?)?;
        Ok(Tensor { buf: y, shape: vec![n, c, oh, ow] })
    }
}

/// `v` padded with zeros to [`RANK`] values.
fn pad6(v: &[usize]) -> [usize; RANK] {
    let mut out = [0; RANK];
    for (o, x) in out.iter_mut().zip(v) {
        *o = *x;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_and_axes_are_checked() {
        assert_eq!(strides(&[2, 3, 4]).ok(), Some(vec![12, 4, 1]));
        assert!(strides(&[2, usize::MAX, 2]).is_err());
        assert_eq!(axis(-1, 3).ok(), Some(2));
        assert!(axis(3, 3).is_err());
        assert!(axis(i64::MIN, 3).is_err());
        assert_eq!(out_len(48, 2, 3, 1).ok(), Some(48));
        assert!(out_len(2, 0, 3, 1).is_err());
        assert_eq!(same_upper(5, 3, 2), (1, 1));
        assert!(pads4(Some(vec![1, -1, 0, 0])).is_err());
        assert!(pair(Some(vec![0, 2]), [1, 1], "x").is_err());
        assert_eq!(pad6(&[1, 2]), [1, 2, 0, 0, 0, 0]);
    }
}
