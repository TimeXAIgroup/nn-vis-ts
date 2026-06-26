// Data-driven 3D renderer for the AudioMNIST 1D-CNN.
// Reads layout.json (layer spec + node positions), renders one cube per node,
// colors cubes by activation, and on hover highlights a node + draws illustrative
// edges to its receptive field in the previous layer.

var Viz = (function () {
    var LAYOUT, layers, nodes, N;
    var posX = [], posY = [], posZ = [], layerNum = [];
    var layerStart = [], layerCount = [];
    var act = [], actRaw = [];
    var isComputed = false;
    var layerHidden = {};
    var gradcamActive = false, cam = null, lastConvIdx = -1;

    var CUBE = 7;
    var FACES_PER_CUBE = 12;

    var container, camera, scene, renderer, controls, stats;
    var edgeObj, edgeMat, highlightBox, waveLine;
    var labelSprites = [];
    var raycaster, mouse = new THREE.Vector2(), hoverId = -1;

    function colorFor(v) {
        var n = Math.max(0, Math.min(99, Math.round(v * 99)));
        return [redLookup[n], greenLookup[n], blueLookup[n]];
    }

    function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

    // Jet colormap for Grad-CAM (0 = blue, 0.5 = green, 1 = red).
    function jet(v) {
        v = clamp01(v);
        var f = 4 * v;
        return [clamp01(Math.min(f - 1.5, -f + 4.5)),
                clamp01(Math.min(f - 0.5, -f + 3.5)),
                clamp01(Math.min(f + 0.5, -f + 2.5))];
    }

    // Grad-CAM value for a node, mapping its display time index onto the cam array.
    function camValueForNode(id, L) {
        var sp = layers[L], t = tc(id).t;
        var frac = sp.displayLen > 1 ? t / (sp.displayLen - 1) : 0;
        return cam[Math.round(frac * (cam.length - 1))];
    }

    function init(layout, containerEl) {
        LAYOUT = layout; layers = layout.layers; nodes = layout.nodes; N = nodes.length;
        for (var i = 0; i < N; i++) {
            posX[i] = nodes[i].x; posY[i] = nodes[i].y; posZ[i] = nodes[i].z;
            layerNum[i] = nodes[i].layerNum;
            act[i] = 0; actRaw[i] = 0;
        }
        // layer offsets (nodes are emitted in layer order)
        for (var li = 0; li < layers.length; li++) { layerStart[li] = -1; layerCount[li] = 0; }
        for (var k = 0; k < N; k++) {
            var L = layerNum[k];
            if (layerStart[L] < 0) layerStart[L] = k;
            layerCount[L]++;
        }
        for (var li2 = 0; li2 < layers.length; li2++) if (layers[li2].type === "conv") lastConvIdx = li2;

        container = containerEl;
        var W = window.innerWidth, H = window.innerHeight;
        camera = new THREE.PerspectiveCamera(60, W / H, 1, 8000);
        camera.position.set(0, layers.length * 95 / 2, 1500);

        scene = new THREE.Scene();
        scene.add(new THREE.AmbientLight(0xffffff, 0.9));
        var sl = new THREE.SpotLight(0xffffff, 0.6); sl.position.set(0, 800, 1500); scene.add(sl);

        buildCubes();
        buildLabels();
        edgeMat = new THREE.LineBasicMaterial({ vertexColors: THREE.VertexColors,
            transparent: true, opacity: 0.85 });
        edgeObj = null;

        highlightBox = new THREE.Mesh(new THREE.BoxGeometry(CUBE + 6, CUBE + 6, CUBE + 6),
            new THREE.MeshBasicMaterial({ color: 0xffff00, wireframe: true }));
        highlightBox.visible = false;
        scene.add(highlightBox);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setClearColor(0x000000, 0);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(W, H);
        container.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, layers.length * 95 / 2, 0);

        try { stats = new Stats(); stats.domElement.style.position = 'absolute'; stats.domElement.style.top = '0';
              container.appendChild(stats.domElement); } catch (e) {}

        raycaster = new THREE.Raycaster();
        renderer.domElement.addEventListener('mousemove', onMouseMove);
        window.addEventListener('resize', onResize);

        updateColors();
        animate();
    }

    // One mesh per layer, so a hidden layer can be made truly transparent
    // (mesh.visible = false) rather than coloured black.
    var layerMeshes = [], layerGeoms = [];
    function buildCubes() {
        var box = new THREE.BoxGeometry(CUBE, CUBE, CUBE);
        var m = new THREE.Matrix4();
        for (var L = 0; L < layers.length; L++) {
            var geom = new THREE.Geometry();
            for (var j = 0; j < layerCount[L]; j++) {
                var id = layerStart[L] + j;
                m.makeTranslation(posX[id], posY[id], posZ[id]);
                geom.merge(box, m);
            }
            var mesh = new THREE.Mesh(geom,
                new THREE.MeshLambertMaterial({ vertexColors: THREE.FaceColors }));
            mesh.frustumCulled = false;
            mesh.userData.layer = L;
            layerGeoms[L] = geom; layerMeshes[L] = mesh;
            scene.add(mesh);
        }
    }

    // Human-readable parameter line for a layer, from its layout spec.
    function paramText(l) {
        if (l.type === "input") return l.length + " samples @ 8 kHz";
        if (l.type === "conv") return l.channels + " filters · kernel " + l.kernel +
            (l.strides > 1 ? " · stride " + l.strides : "") + " → " + l.length + "×" + l.channels;
        if (l.type === "pool") return "max-pool /" + l.pool + " → " + l.length + "×" + l.channels;
        if (l.name === "output") return l.length + " classes · softmax";
        if (l.type === "dense") return l.length + " units · ReLU";
        return "";
    }

    // A floating two-line text label placed just to the right of each layer.
    function buildLabels() {
        var lines = layers.map(function (l) { return [l.label, paramText(l)]; });
        var probe = document.createElement("canvas").getContext("2d");
        var titleFont = "bold 26px sans-serif", paramFont = "22px sans-serif";
        var maxW = 0;
        lines.forEach(function (ln) {
            probe.font = titleFont; maxW = Math.max(maxW, probe.measureText(ln[0]).width);
            probe.font = paramFont; maxW = Math.max(maxW, probe.measureText(ln[1]).width);
        });
        var cw = Math.ceil(maxW) + 24, ch = 76, sx = 0.7;

        for (var L = 0; L < layers.length; L++) {
            var cv = document.createElement("canvas"); cv.width = cw; cv.height = ch;
            var isInput = layers[L].type === "input";
            var ctx = cv.getContext("2d");
            ctx.textBaseline = "top";
            var tx = isInput ? cw / 2 : cw - 4;   // input: centered; others: right-aligned (left of layer)
            ctx.textAlign = isInput ? "center" : "right";
            ctx.font = titleFont; ctx.fillStyle = "#7cd3ff"; ctx.fillText(lines[L][0], tx, 4);
            ctx.font = paramFont; ctx.fillStyle = "#cfcfcf"; ctx.fillText(lines[L][1], tx, 40);

            var tex = new THREE.Texture(cv); tex.minFilter = THREE.LinearFilter; tex.needsUpdate = true;
            var sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
            var worldW = cw * sx, worldH = ch * sx;
            if (isInput) {
                // centered, below the waveform line (which sits at posY[0] - 38, amp 30)
                sprite.position.set(0, posY[layerStart[L]] - 95, 0);
            } else {
                var xMin = posX[layerStart[L]];
                sprite.position.set(xMin - 40 - worldW / 2, posY[layerStart[L]], 0);
            }
            sprite.scale.set(worldW, worldH, 1);
            labelSprites[L] = sprite;
            scene.add(sprite);
        }
    }

    function updateColors() {
        for (var L = 0; L < layers.length; L++) {
            var mesh = layerMeshes[L];
            if (layerHidden[L]) { mesh.visible = false; continue; }
            mesh.visible = true;
            var faces = layerGeoms[L].faces, start = layerStart[L];
            for (var f = 0; f < faces.length; f++) {
                var id = start + Math.floor(f / FACES_PER_CUBE);
                var c;
                if (gradcamActive && (L === 0 || L === lastConvIdx)) { c = jet(camValueForNode(id, L)); }
                else if (gradcamActive) { c = colorFor(act[id] * 0.25); }  // dim other layers
                else if (isComputed) { c = colorFor(act[id]); }
                else { c = [0.06, 0.06, 0.06]; }
                faces[f].color.setRGB(c[0], c[1], c[2]);
            }
            layerGeoms[L].colorsNeedUpdate = true;
        }
    }

    // within-layer index -> {t, c}
    function tc(id) {
        var L = layerNum[id], C = layers[L].channels;
        var w = id - layerStart[L];
        return { L: L, t: Math.floor(w / C), c: w % C };
    }

    function buildEdges(id) {
        var g = new THREE.Geometry();
        var info = tc(id), L = info.L;
        if (L <= 0) { setEdges(g); return; }
        var prev = L - 1, ps = layers[prev], cur = layers[L];
        var prevStart = layerStart[prev], Cp = ps.channels, Dp = ps.displayLen;

        function link(pid) {
            g.vertices.push(new THREE.Vector3(posX[pid], posY[pid], posZ[pid]));
            g.vertices.push(new THREE.Vector3(posX[id], posY[id], posZ[id]));
            var col = isComputed ? colorFor(act[pid]) : [0.5, 0.5, 0.5];
            var thc = new THREE.Color(col[0], col[1], col[2]);
            g.colors.push(thc, thc);
        }

        if (cur.type === "dense") {
            for (var p = 0; p < layerCount[prev]; p++) link(prevStart + p);
        } else {
            var Dk = cur.displayLen;
            var center = Dk > 1 ? Math.round((info.t / (Dk - 1)) * (Dp - 1)) : 0;
            var win = 3;
            for (var tp = Math.max(0, center - win); tp <= Math.min(Dp - 1, center + win); tp++) {
                for (var cp = 0; cp < Cp; cp++) {
                    if (cur.type === "pool" && cp !== info.c) continue; // pooling keeps channels
                    link(prevStart + tp * Cp + cp);
                }
            }
        }
        setEdges(g);
    }

    function setEdges(g) {
        if (edgeObj) { scene.remove(edgeObj); edgeObj.geometry.dispose(); edgeObj = null; }
        if (g.vertices.length === 0) return;
        edgeObj = new THREE.Line(g, edgeMat, THREE.LinePieces);
        edgeObj.frustumCulled = false;
        scene.add(edgeObj);
    }

    function onMouseMove(e) {
        mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        var visible = [];
        for (var L = 0; L < layerMeshes.length; L++) if (!layerHidden[L]) visible.push(layerMeshes[L]);
        var hits = raycaster.intersectObjects(visible);
        var id = -1;
        if (hits.length) {
            var obj = hits[0].object, lyr = obj.userData.layer;
            id = layerStart[lyr] + Math.floor(hits[0].faceIndex / FACES_PER_CUBE);
        }
        if (id < 0) { highlightBox.visible = false; hoverId = -1; setEdges(new THREE.Geometry());
                      if (onHover) onHover(null); return; }
        if (id === hoverId) return;
        hoverId = id;
        highlightBox.position.set(posX[id], posY[id], posZ[id]);
        highlightBox.visible = true;
        buildEdges(id);
        if (onHover) onHover(describe(id));
    }

    function describe(id) {
        var info = tc(id), spec = layers[info.L];
        return {
            name: spec.name, layer: spec.label, type: spec.type,
            t: info.t, c: info.c, channels: spec.channels, displayLen: spec.displayLen,
            raw: actRaw[id], norm: act[id],
        };
    }

    var onHover = null;
    function setHoverCallback(cb) { onHover = cb; }

    // perLayer: array aligned to layers, each {values, raw} length = displayLen*channels
    function setActivations(perLayer) {
        for (var L = 0; L < layers.length; L++) {
            var start = layerStart[L], pl = perLayer[L];
            for (var j = 0; j < layerCount[L]; j++) {
                act[start + j] = pl.values[j];
                actRaw[start + j] = pl.raw[j];
            }
        }
        isComputed = true;
        updateColors();
        if (hoverId >= 0) buildEdges(hoverId);
    }

    // Draw the actual waveform as a 3D line just beneath the input cube row,
    // time-aligned with it, so the link between samples and signal is obvious.
    function setWaveform(wave) {
        if (waveLine) { scene.remove(waveLine); waveLine.geometry.dispose(); waveLine = null; }
        var n0 = layerCount[0];
        var xMin = posX[0], xMax = posX[n0 - 1];
        var baseY = posY[0] - 38, amp = 30;
        var npts = Math.min(wave.length, 700);
        var g = new THREE.Geometry();
        for (var i = 0; i < npts; i++) {
            var frac = npts > 1 ? i / (npts - 1) : 0;
            var x = xMin + frac * (xMax - xMin);
            var s = wave[Math.round(frac * (wave.length - 1))] || 0;
            g.vertices.push(new THREE.Vector3(x, baseY + s * amp, 0));
        }
        waveLine = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x33cc88 }));
        waveLine.frustumCulled = false;
        scene.add(waveLine);
    }

    function showGradcam(camArr) {
        cam = camArr; gradcamActive = true; updateColors();
    }
    function clearGradcam() {
        if (!gradcamActive) return;
        gradcamActive = false; cam = null; updateColors();
    }
    // Sample cam at a fraction 0..1 along the input time axis (for the waveform overlay).
    function camAt(frac) {
        if (!cam) return 0;
        return cam[Math.round(clamp01(frac) * (cam.length - 1))];
    }

    function setLayerHidden(layerName, hidden) {
        for (var L = 0; L < layers.length; L++) if (layers[L].name === layerName) {
            layerHidden[L] = hidden;
            if (labelSprites[L]) labelSprites[L].visible = !hidden;
        }
        updateColors();
    }

    function onResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
        if (stats) stats.update();
    }

    return { init: init, setActivations: setActivations, setHoverCallback: setHoverCallback,
             setLayerHidden: setLayerHidden, showGradcam: showGradcam, clearGradcam: clearGradcam,
             jet: jet, camAt: camAt, setWaveform: setWaveform };
})();
