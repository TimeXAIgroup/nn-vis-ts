// Inference for the AudioMNIST 1D-CNN, in the browser via TensorFlow.js.
//
// Loads the converted Keras model, builds a multi-output model exposing every
// visualized layer, runs one sample, and lays the activations out in the exact
// node order produced by model/export_web.py (per layer: time-major, then channel,
// with the time axis evenly subsampled to layer.displayLen).

var Infer = (function () {
    var baseModel = null;   // tf.LayersModel
    var multiModel = null;  // model with one output per visualized layer
    var layersSpec = null;  // layout.layers

    // Evenly spaced display indices into an actual axis of length L.
    function displayIndices(L, D) {
        var idx = new Array(D);
        if (D >= L) { for (var i = 0; i < D; i++) idx[i] = Math.min(i, L - 1); return idx; }
        for (var t = 0; t < D; t++) idx[t] = Math.round(t * (L - 1) / (D - 1));
        return idx;
    }

    var filterCache = {};  // layerName -> {K, inC, outC, data:Float32Array}
    var convToLast = null; // model: input -> last conv output
    var tailModel = null;  // model: last conv output -> penultimate features
    var outW = null, outB = null; // output Dense weights (for pre-softmax logits)
    var lastConvName = null;

    async function load(modelUrl, spec) {
        layersSpec = spec;
        baseModel = await tf.loadLayersModel(modelUrl);
        var outputs = layersSpec.map(function (l) { return baseModel.getLayer(l.name).output; });
        multiModel = tf.model({ inputs: baseModel.inputs, outputs: outputs });

        // Cache conv kernels for filter visualization.
        layersSpec.forEach(function (l) {
            if (l.type !== "conv") return;
            var w = baseModel.getLayer(l.name).getWeights()[0]; // [K, inC, outC]
            var s = w.shape;
            filterCache[l.name] = { K: s[0], inC: s[1], outC: s[2], data: w.dataSync().slice() };
        });

        // Split the network at the last conv layer for Grad-CAM.
        var convs = layersSpec.filter(function (l) { return l.type === "conv"; });
        lastConvName = convs[convs.length - 1].name;
        var convIdx = baseModel.layers.findIndex(function (l) { return l.name === lastConvName; });
        var lastConv = baseModel.layers[convIdx];
        convToLast = tf.model({ inputs: baseModel.inputs, outputs: lastConv.output });

        // Tail from the last conv output up to the PENULTIMATE layer (i.e. the
        // input to the final softmax). Grad-CAM then differentiates the raw
        // class logit = penult @ Wout + bout, not the softmax probability.
        var last = baseModel.layers.length - 1;
        var ti = tf.input({ shape: lastConv.outputShape.slice(1) });
        var y = ti;
        for (var i = convIdx + 1; i < last; i++) y = baseModel.layers[i].apply(y);
        tailModel = tf.model({ inputs: ti, outputs: y });           // -> penultimate features
        var ow = baseModel.layers[last].getWeights();               // [kernel, bias] of output Dense
        outW = ow[0]; outB = ow[1];                                 // kept (module-level, not in tidy)
        return true;
    }

    // 1D Grad-CAM w.r.t. the last conv layer for a target class.
    // Returns a normalized Float32Array of length = last-conv time steps.
    function gradcam(classIdx, waveform, inputLen) {
        return tf.tidy(function () {
            var x = tf.tensor(waveform, [1, inputLen, 1]);
            var A = convToLast.predict(x);                       // [1, T, C]
            var scoreFn = function (a) {
                var penult = tailModel.apply(a);                  // [1, units]
                var logits = penult.matMul(outW).add(outB);       // pre-softmax  [1, classes]
                return logits.slice([0, classIdx], [1, 1]).sum();
            };
            var grads = tf.grad(scoreFn)(A);                     // [1, T, C]
            var alpha = grads.mean(1);                           // [1, C]  (avg over time)
            var cam = A.mul(alpha.expandDims(1)).sum(2).squeeze(); // [T]
            cam = cam.relu();
            cam = cam.div(cam.max().add(1e-6));
            return Float32Array.from(cam.dataSync());
        });
    }

    // Learned 1D filter for one output channel, averaged over input channels
    // (the net temporal template the channel responds to). Returns Float32Array(K).
    function getFilter(layerName, outChannel) {
        var f = filterCache[layerName];
        if (!f) return null;
        var out = new Float32Array(f.K);
        for (var k = 0; k < f.K; k++) {
            var sum = 0;
            for (var i = 0; i < f.inC; i++) sum += f.data[k * f.inC * f.outC + i * f.outC + outChannel];
            out[k] = sum / f.inC;
        }
        return out;
    }

    // waveform: Float32Array length inputLen. Returns:
    //   { perLayer: [{values:Float32Array(displayLen*channels normalized 0..1),
    //                 raw:Float32Array(same, unnormalized)}],
    //     probs: Float32Array(10) }
    function run(waveform, inputLen) {
        return tf.tidy(function () {
            var x = tf.tensor(waveform, [1, inputLen, 1]);
            var outs = multiModel.predict(x);
            if (!Array.isArray(outs)) outs = [outs];

            var perLayer = [];
            var probs = null;
            for (var li = 0; li < layersSpec.length; li++) {
                var spec = layersSpec[li];
                var data = outs[li].dataSync();      // flat, shape (length, channels) or (units)
                var L = spec.length, C = spec.channels, D = spec.displayLen;
                var idx = displayIndices(L, D);

                var raw = new Float32Array(D * C);
                var k = 0;
                for (var t = 0; t < D; t++) {
                    var at = idx[t];
                    for (var c = 0; c < C; c++) raw[k++] = data[at * C + c];
                }

                // Normalize for coloring.
                var values = new Float32Array(D * C);
                if (spec.type === "input") {
                    for (var i = 0; i < raw.length; i++) values[i] = (raw[i] + 1) / 2; // [-1,1]->[0,1]
                } else {
                    var mx = 0;
                    for (var j = 0; j < raw.length; j++) if (raw[j] > mx) mx = raw[j];
                    if (mx <= 0) mx = 1;
                    for (var m = 0; m < raw.length; m++) values[m] = Math.max(0, raw[m]) / mx;
                }
                perLayer.push({ values: values, raw: raw });
                if (spec.name === "output") probs = raw;
            }
            return { perLayer: perLayer, probs: probs };
        });
    }

    return { load: load, run: run, getFilter: getFilter, gradcam: gradcam,
             lastConvName: function () { return lastConvName; } };
})();
