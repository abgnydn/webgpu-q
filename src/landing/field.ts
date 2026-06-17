// Full-bleed hero background: a living quantum wavefunction. A WebGL fragment
// shader sums a handful of superposed atomic orbitals (s/p/d), each evolving
// its own complex phase e^{-iEt}; the pixel color is the phase (mapped across
// the visible emission spectrum) modulated by the probability density |ψ|².
// The cursor adds a moving orbital — perturbing the field like a measurement.
// Cheap, GPU-driven, full-screen; degrades to the CSS gradient if WebGL is absent.

const VERT = `
attribute vec2 a_pos;
void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
uniform vec2  u_res;
uniform float u_time;
uniform vec2  u_mouse;
uniform float u_reduce;

// Iridescent spectral palette (deep blue → cyan → violet → amber).
vec3 pal(float t){ return 0.5 + 0.5*cos(6.28318*(vec3(0.0,0.36,0.61)+t)); }

// One hydrogenic-ish orbital's complex amplitude at point p.
vec2 orbital(vec2 p, vec2 c, float l, float a0, float ph){
  vec2 d = p - c;
  float r = length(d);
  float th = atan(d.y, d.x);
  float rad = exp(-r*r*0.42);
  float ang = (l < 0.5) ? 1.0 : cos(l*th + a0);
  float amp = rad * ang;
  return amp * vec2(cos(ph), sin(ph));
}

void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*u_res) / min(u_res.x, u_res.y);
  uv *= 3.0;
  float t = u_time * (u_reduce > 0.5 ? 0.0 : 1.0);

  vec2 psi = vec2(0.0);
  psi += orbital(uv, vec2(-1.45,  0.75), 2.0, 0.0,  t*0.70);
  psi += orbital(uv, vec2( 1.55, -0.45), 1.0, 1.57, -t*0.52 + 1.0);
  psi += orbital(uv, vec2( 0.20,  1.05), 3.0, 0.50, t*0.41 + 2.0);
  psi += orbital(uv, vec2(-0.65, -1.10), 0.0, 0.0, -t*0.61);
  psi += orbital(uv, vec2( 1.05,  1.20), 2.0, 2.0,  t*0.55 + 0.5);
  vec2 m = u_mouse * 3.0;
  psi += 1.25 * orbital(uv, m, 2.0, t, t*0.9);

  float inten = dot(psi, psi);
  float phase = atan(psi.y, psi.x) / 6.28318 + 0.5;
  vec3 col = pal(phase + 0.04*t) * pow(inten, 0.55);
  col = col / (1.0 + col);                 // tone-map (no blowout)
  float vig = smoothstep(2.6, 0.25, length(uv));
  col *= 0.55 + 0.45*vig;
  col += vec3(0.018, 0.024, 0.05);         // deep base tint
  gl_FragColor = vec4(col, 1.0);
}`;

function compile(gl: WebGLRenderingContext, type: number, src: string): WebGLShader | null {
  const s = gl.createShader(type);
  if (!s) return null;
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { gl.deleteShader(s); return null; }
  return s;
}

export function initField(canvas: HTMLCanvasElement): void {
  const gl = (canvas.getContext("webgl", { antialias: false, alpha: false }) ||
    canvas.getContext("experimental-webgl")) as WebGLRenderingContext | null;
  if (!gl) { canvas.style.display = "none"; return; }

  const vs = compile(gl, gl.VERTEX_SHADER, VERT);
  const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
  if (!vs || !fs) { canvas.style.display = "none"; return; }
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { canvas.style.display = "none"; return; }
  gl.useProgram(prog);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "a_pos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  const uRes = gl.getUniformLocation(prog, "u_res");
  const uTime = gl.getUniformLocation(prog, "u_time");
  const uMouse = gl.getUniformLocation(prog, "u_mouse");
  const uReduce = gl.getUniformLocation(prog, "u_reduce");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : 0;
  const DPR = Math.min(window.devicePixelRatio || 1, 1.75);

  const start = performance.now();
  let W = 0, H = 0, mx = 5, my = 5, tx = 5, ty = 5;

  const draw = (): void => {
    mx += (tx - mx) * 0.06; my += (ty - my) * 0.06;
    gl.uniform2f(uRes, canvas.width, canvas.height);
    gl.uniform1f(uTime, (performance.now() - start) / 1000);
    gl.uniform2f(uMouse, mx, my);
    gl.uniform1f(uReduce, reduce);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };

  // Desktop animates; mobile/reduced-motion get a crisp STATIC frame — a
  // full-screen fragment shader is a real battery/heat cost on phones.
  const mobile = matchMedia("(pointer: coarse)").matches || Math.min(window.innerWidth, window.innerHeight) < 640;
  const animate = !reduce && !mobile;

  const resize = (): void => {
    W = Math.floor(canvas.clientWidth * DPR); H = Math.floor(canvas.clientHeight * DPR);
    canvas.width = Math.max(1, W); canvas.height = Math.max(1, H);
    gl.viewport(0, 0, canvas.width, canvas.height);
    if (!animate) draw();   // keep the static frame correct across resizes
  };
  resize();
  window.addEventListener("resize", resize);

  window.addEventListener("pointermove", (e) => {
    const min = Math.min(canvas.clientWidth, canvas.clientHeight);
    tx = (e.clientX - canvas.clientWidth * 0.5) / min;
    ty = -(e.clientY - canvas.clientHeight * 0.5) / min;
  }, { passive: true });

  // Throttle (rAF fires at the display refresh rate — 120Hz on many panels) and
  // pause entirely when the tab is hidden, so the always-on background isn't a
  // CPU/battery drain. A slowly-evolving field looks identical at 36fps.
  const minDt = 1000 / 36;
  let last = -1e9, raf = 0, running = false;
  const loop = (now: number): void => {
    if (!running) return;
    if (now - last >= minDt) { last = now; draw(); }
    raf = requestAnimationFrame(loop);
  };
  const startLoop = (): void => { if (!running) { running = true; last = -1e9; raf = requestAnimationFrame(loop); } };
  const stopLoop = (): void => { running = false; cancelAnimationFrame(raf); };

  if (!animate) { draw(); }
  else {
    startLoop();
    document.addEventListener("visibilitychange", () => { if (document.hidden) stopLoop(); else startLoop(); });
  }
}
