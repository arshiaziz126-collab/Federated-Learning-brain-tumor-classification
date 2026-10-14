// NeuroFed AI - the trained CNN, run in plain JavaScript.
//
// Preprocessing reproduces Pillow exactly (RGB -> "L" conversion, then the BILINEAR resize
// with antialiasing that Image.resize uses), so a decoded image gives the same 128x128 input
// as backend/preprocessing.py. The network then repeats the Keras layers one by one with the
// exported weights (tools/export_web_model.py). Nothing leaves the browser.

export const CLASS_NAMES_DEFAULT = ["glioma", "meningioma", "notumor", "pituitary"];

// ---------- preprocessing (Pillow-compatible) ----------
// rgba: Uint8ClampedArray from canvas getImageData (RGBA). Returns Uint8Array gray (w*h).
export function rgbaToL(rgba, w, h) {
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    // Pillow: L = (R*19595 + G*38470 + B*7471 + 0x8000) >> 16
    out[i] = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >>> 16;
  }
  return out;
}

const PRECISION_BITS = 32 - 8 - 2;

function bilinear(x) {
  if (x < 0) x = -x;
  return x < 1 ? 1 - x : 0;
}

function precomputeCoeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = 1.0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const ss = 1.0 / filterscale;
  const k = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let ww = 0.0;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    for (let x = 0; x < xmax; x++) {
      const w = bilinear((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      // normalize_coeffs_8bpc: C cast truncates toward zero after adding +/-0.5
      kk[xx * ksize + x] = Math.trunc(v * (1 << PRECISION_BITS) + (v < 0 ? -0.5 : 0.5));
    }
    for (let x = xmax; x < ksize; x++) kk[xx * ksize + x] = 0;
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

function clip8(v) {
  if (v >= 2 ** (PRECISION_BITS + 8)) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / (1 << PRECISION_BITS));
}

// Pillow Image.resize((outW, outH), Image.BILINEAR) for a single-channel 8-bit image.
export function resizeLikePillow(src, w, h, outW, outH) {
  let cur = src, cw = w;
  if (outW !== w) {
    const { ksize, bounds, kk } = precomputeCoeffs(w, outW);
    const tmp = new Uint8Array(outW * h);
    for (let y = 0; y < h; y++) {
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1];
        let s = 1 << (PRECISION_BITS - 1);
        for (let x = 0; x < xmax; x++) s += cur[y * cw + x + xmin] * kk[xx * ksize + x];
        tmp[y * outW + xx] = clip8(s);
      }
    }
    cur = tmp; cw = outW;
  }
  if (outH !== h) {
    const { ksize, bounds, kk } = precomputeCoeffs(h, outH);
    const tmp = new Uint8Array(cw * outH);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1];
      for (let x = 0; x < cw; x++) {
        let s = 1 << (PRECISION_BITS - 1);
        for (let y = 0; y < ymax; y++) s += cur[(y + ymin) * cw + x] * kk[yy * ksize + y];
        tmp[yy * cw + x] = clip8(s);
      }
    }
    cur = tmp;
  }
  return cur;
}

// ---------- network ----------
export async function loadModel(baseUrl, name) {
  const [manifest, buf] = await Promise.all([
    fetch(`${baseUrl}/${name}.json`).then((r) => { if (!r.ok) throw new Error(`Model manifest ${name} not found`); return r.json(); }),
    fetch(`${baseUrl}/${name}.bin`).then((r) => { if (!r.ok) throw new Error(`Model weights ${name} not found`); return r.arrayBuffer(); }),
  ]);
  return buildModel(manifest, new Float32Array(buf));
}

export function buildModel(manifest, data) {
  const layers = manifest.layers.map((l) => ({
    ...l,
    w: (l.weights || []).map((wt) => ({ shape: wt.shape, data: data.subarray(wt.offset, wt.offset + wt.length) })),
  }));
  return { manifest, layers, inputShape: manifest.inputShape };
}

function conv3x3Same(x, H, W, Cin, kernel, Cout) {
  // x: HWC float32; kernel: [3,3,Cin,Cout]
  const out = new Float32Array(H * W * Cout);
  for (let y = 0; y < H; y++) {
    for (let xx = 0; xx < W; xx++) {
      const o = (y * W + xx) * Cout;
      for (let dy = 0; dy < 3; dy++) {
        const iy = y + dy - 1;
        if (iy < 0 || iy >= H) continue;
        for (let dx = 0; dx < 3; dx++) {
          const ix = xx + dx - 1;
          if (ix < 0 || ix >= W) continue;
          const ib = (iy * W + ix) * Cin;
          const kb = (dy * 3 + dx) * Cin * Cout;
          for (let ci = 0; ci < Cin; ci++) {
            const v = x[ib + ci];
            if (v === 0) continue;
            const kr = kb + ci * Cout;
            for (let co = 0; co < Cout; co++) out[o + co] += v * kernel[kr + co];
          }
        }
      }
    }
  }
  return out;
}

function groupNorm(x, H, W, C, groups, eps, gamma, beta) {
  const per = C / groups, n = H * W * per;
  for (let g = 0; g < groups; g++) {
    const c0 = g * per;
    let mean = 0;
    for (let p = 0; p < H * W; p++) for (let c = 0; c < per; c++) mean += x[p * C + c0 + c];
    mean /= n;
    let v = 0;
    for (let p = 0; p < H * W; p++) for (let c = 0; c < per; c++) { const d = x[p * C + c0 + c] - mean; v += d * d; }
    v /= n;
    const inv = 1 / Math.sqrt(v + eps);
    for (let p = 0; p < H * W; p++) for (let c = 0; c < per; c++) {
      const i = p * C + c0 + c;
      x[i] = (x[i] - mean) * inv * gamma[c0 + c] + beta[c0 + c];
    }
  }
  return x;
}

function maxPool2(x, H, W, C) {
  const h = Math.floor(H / 2), w = Math.floor(W / 2);
  const out = new Float32Array(h * w * C);
  for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) for (let c = 0; c < C; c++) {
    const a = x[((2 * y) * W + 2 * xx) * C + c], b = x[((2 * y) * W + 2 * xx + 1) * C + c];
    const d = x[((2 * y + 1) * W + 2 * xx) * C + c], e = x[((2 * y + 1) * W + 2 * xx + 1) * C + c];
    out[(y * w + xx) * C + c] = Math.max(a, b, d, e);
  }
  return { x: out, H: h, W: w };
}

// gray: Uint8Array of length H*W (already 128x128). Returns probabilities (Float64Array).
export function predictProbs(model, gray) {
  let [H, W, C] = model.inputShape;
  let x = new Float32Array(H * W * C);
  for (let i = 0; i < gray.length; i++) x[i] = gray[i];
  let vec = null;
  for (const l of model.layers) {
    switch (l.type) {
      case "Rescaling":
        for (let i = 0; i < x.length; i++) x[i] = x[i] * l.scale + l.offset;
        break;
      case "Conv2D": {
        const [, , cin, cout] = l.w[0].shape;
        x = conv3x3Same(x, H, W, cin, l.w[0].data, cout);
        if (l.useBias) for (let i = 0; i < x.length; i++) x[i] += l.w[1].data[i % cout];
        C = cout;
        break;
      }
      case "GroupNormalization":
        x = groupNorm(x, H, W, C, l.groups, l.epsilon, l.w[0].data, l.w[1].data);
        break;
      case "ReLU":
        if (vec) { for (let i = 0; i < vec.length; i++) if (vec[i] < 0) vec[i] = 0; }
        else for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
        break;
      case "MaxPooling2D": {
        const r = maxPool2(x, H, W, C);
        x = r.x; H = r.H; W = r.W;
        break;
      }
      case "GlobalAveragePooling2D": {
        vec = new Float64Array(C);
        for (let p = 0; p < H * W; p++) for (let c = 0; c < C; c++) vec[c] += x[p * C + c];
        for (let c = 0; c < C; c++) vec[c] /= H * W;
        break;
      }
      case "Dense": {
        const [nin, nout] = l.w[0].shape, k = l.w[0].data, b = l.w[1].data;
        const out = new Float64Array(nout);
        for (let o = 0; o < nout; o++) {
          let s = b[o];
          for (let i = 0; i < nin; i++) s += vec[i] * k[i * nout + o];
          out[o] = s;
        }
        if (l.activation === "relu") for (let o = 0; o < nout; o++) if (out[o] < 0) out[o] = 0;
        if (l.activation === "softmax") {
          const m = Math.max(...out);
          let z = 0;
          for (let o = 0; o < nout; o++) { out[o] = Math.exp(out[o] - m); z += out[o]; }
          for (let o = 0; o < nout; o++) out[o] /= z;
        }
        vec = out;
        break;
      }
      default:
        throw new Error(`Unsupported layer ${l.type}`);
    }
  }
  return vec;
}

// Same fields and thresholds as backend/inference.py
export function summarise(probs, classNames, { lowConfidence = 0.70, lowMargin = 0.15 } = {}) {
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
  const top = order[0], second = order[1];
  const confidence = probs[top];
  const margin = second !== undefined ? probs[top] - probs[second] : confidence;
  const negative = classNames.find((n) => ["notumor", "no_tumor", "no tumour", "no_tumour", "normal", "healthy"]
    .includes(n.toLowerCase())) || null;
  const probabilities = {};
  for (const i of order) probabilities[classNames[i]] = probs[i];
  return {
    predicted_class: classNames[top],
    confidence,
    margin_to_runner_up: margin,
    runner_up: second !== undefined ? classNames[second] : null,
    probabilities,
    low_confidence: confidence < lowConfidence || margin < lowMargin,
    negative_class: negative,
    screen_positive: negative ? classNames[top] !== negative : null,
  };
}
