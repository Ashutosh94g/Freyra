"""Freyra Pose Editor -- interactive stick-figure canvas for pose guidance.

Provides a draggable 17-keypoint skeleton editor rendered on an HTML5 canvas.
The output is a white-on-black stick figure image suitable for PyraCanny ControlNet.
"""

import gradio as gr
import json

KEYPOINT_NAMES = [
    'nose', 'neck',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'right_eye', 'left_eye',
    'head_top',
]

SKELETON_CONNECTIONS = [
    ('head_top', 'nose'), ('nose', 'right_eye'), ('nose', 'left_eye'),
    ('nose', 'neck'),
    ('neck', 'right_shoulder'), ('neck', 'left_shoulder'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('neck', 'right_hip'), ('neck', 'left_hip'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'left_hip'),
]

DEFAULT_KEYPOINTS = {
    'head_top':        (0.50, 0.04),
    'nose':            (0.50, 0.10),
    'right_eye':       (0.47, 0.08),
    'left_eye':        (0.53, 0.08),
    'neck':            (0.50, 0.18),
    'right_shoulder':  (0.38, 0.22),
    'left_shoulder':   (0.62, 0.22),
    'right_elbow':     (0.30, 0.38),
    'left_elbow':      (0.70, 0.38),
    'right_wrist':     (0.26, 0.52),
    'left_wrist':      (0.74, 0.52),
    'right_hip':       (0.42, 0.52),
    'left_hip':        (0.58, 0.52),
    'right_knee':      (0.40, 0.72),
    'left_knee':       (0.60, 0.72),
    'right_ankle':     (0.40, 0.90),
    'left_ankle':      (0.60, 0.90),
}

LIMB_COLORS = {
    ('head_top', 'nose'): '#FFD700',
    ('nose', 'right_eye'): '#FFD700',
    ('nose', 'left_eye'): '#FFD700',
    ('nose', 'neck'): '#FFD700',
    ('neck', 'right_shoulder'): '#FF6B6B',
    ('right_shoulder', 'right_elbow'): '#FF6B6B',
    ('right_elbow', 'right_wrist'): '#FF6B6B',
    ('neck', 'left_shoulder'): '#4ECDC4',
    ('left_shoulder', 'left_elbow'): '#4ECDC4',
    ('left_elbow', 'left_wrist'): '#4ECDC4',
    ('neck', 'right_hip'): '#FF6B6B',
    ('right_hip', 'right_knee'): '#FF6B6B',
    ('right_knee', 'right_ankle'): '#FF6B6B',
    ('neck', 'left_hip'): '#4ECDC4',
    ('left_hip', 'left_knee'): '#4ECDC4',
    ('left_knee', 'left_ankle'): '#4ECDC4',
    ('right_hip', 'left_hip'): '#C0C0C0',
}


def _build_pose_editor_html(editor_id: str) -> str:
    kp_json = json.dumps(DEFAULT_KEYPOINTS)
    conn_json = json.dumps(SKELETON_CONNECTIONS)
    colors_json = json.dumps({f"{a}_{b}": c for (a, b), c in LIMB_COLORS.items()})

    return f"""
<div id="{editor_id}_wrap" style="position:relative;display:inline-block;">
  <canvas id="{editor_id}" width="384" height="512"
    style="background:#000;border:1px solid #444;border-radius:8px;cursor:crosshair;
           max-width:100%;touch-action:none;"></canvas>
  <div style="display:flex;gap:8px;margin-top:6px;">
    <button type="button" onclick="freyra_pose_reset('{editor_id}')"
      style="background:#333;color:#ccc;border:1px solid #555;border-radius:4px;
             padding:4px 12px;cursor:pointer;font-size:12px;">Reset Pose</button>
    <button type="button" onclick="freyra_pose_export('{editor_id}')"
      style="background:#c4852e;color:#fff;border:none;border-radius:4px;
             padding:4px 12px;cursor:pointer;font-size:12px;">Apply Pose</button>
  </div>
</div>
<script>
(function() {{
  var eid = '{editor_id}';
  var canvas = document.getElementById(eid);
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var defaultKP = {kp_json};
  var connections = {conn_json};
  var limbColors = {colors_json};
  var kp = JSON.parse(JSON.stringify(defaultKP));
  var dragging = null;
  var JOINT_RADIUS = 7;

  function toPixel(name) {{
    return [kp[name][0] * W, kp[name][1] * H];
  }}

  function draw() {{
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);

    for (var i = 0; i < connections.length; i++) {{
      var a = connections[i][0], b = connections[i][1];
      var pa = toPixel(a), pb = toPixel(b);
      var ckey = a + '_' + b;
      ctx.strokeStyle = limbColors[ckey] || '#FFFFFF';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
      ctx.stroke();
    }}

    var names = Object.keys(kp);
    for (var j = 0; j < names.length; j++) {{
      var p = toPixel(names[j]);
      ctx.fillStyle = '#FFFFFF';
      ctx.beginPath();
      ctx.arc(p[0], p[1], JOINT_RADIUS, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1;
      ctx.stroke();
    }}
  }}

  function getPos(e) {{
    var rect = canvas.getBoundingClientRect();
    var scaleX = W / rect.width, scaleY = H / rect.height;
    var clientX, clientY;
    if (e.touches && e.touches.length > 0) {{
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }} else {{
      clientX = e.clientX;
      clientY = e.clientY;
    }}
    return [(clientX - rect.left) * scaleX, (clientY - rect.top) * scaleY];
  }}

  function findJoint(px, py) {{
    var names = Object.keys(kp);
    var best = null, bestDist = JOINT_RADIUS * 2.5;
    for (var i = 0; i < names.length; i++) {{
      var p = toPixel(names[i]);
      var d = Math.sqrt((px - p[0]) ** 2 + (py - p[1]) ** 2);
      if (d < bestDist) {{ best = names[i]; bestDist = d; }}
    }}
    return best;
  }}

  function onDown(e) {{
    e.preventDefault();
    var pos = getPos(e);
    dragging = findJoint(pos[0], pos[1]);
  }}
  function onMove(e) {{
    if (!dragging) return;
    e.preventDefault();
    var pos = getPos(e);
    kp[dragging] = [
      Math.max(0, Math.min(1, pos[0] / W)),
      Math.max(0, Math.min(1, pos[1] / H))
    ];
    draw();
  }}
  function onUp(e) {{ dragging = null; }}

  canvas.addEventListener('mousedown', onDown);
  canvas.addEventListener('mousemove', onMove);
  canvas.addEventListener('mouseup', onUp);
  canvas.addEventListener('mouseleave', onUp);
  canvas.addEventListener('touchstart', onDown, {{passive: false}});
  canvas.addEventListener('touchmove', onMove, {{passive: false}});
  canvas.addEventListener('touchend', onUp);

  window['freyra_pose_kp_' + eid] = kp;
  window['freyra_pose_default_' + eid] = defaultKP;
  window['freyra_pose_draw_' + eid] = draw;

  window.freyra_pose_reset = function(id) {{
    var dk = window['freyra_pose_default_' + id];
    var ckp = window['freyra_pose_kp_' + id];
    var names = Object.keys(dk);
    for (var i = 0; i < names.length; i++) {{
      ckp[names[i]] = [dk[names[i]][0], dk[names[i]][1]];
    }}
    window['freyra_pose_draw_' + id]();
  }};

  window.freyra_pose_export = function(id) {{
    var c = document.getElementById(id);
    if (!c) return;
    var tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = c.width;
    tmpCanvas.height = c.height;
    var tmpCtx = tmpCanvas.getContext('2d');
    var ckp = window['freyra_pose_kp_' + id];
    var w = tmpCanvas.width, h = tmpCanvas.height;

    tmpCtx.fillStyle = '#000';
    tmpCtx.fillRect(0, 0, w, h);
    for (var i = 0; i < connections.length; i++) {{
      var a = connections[i][0], b = connections[i][1];
      var pa = [ckp[a][0]*w, ckp[a][1]*h];
      var pb = [ckp[b][0]*w, ckp[b][1]*h];
      tmpCtx.strokeStyle = '#FFFFFF';
      tmpCtx.lineWidth = 4;
      tmpCtx.beginPath();
      tmpCtx.moveTo(pa[0], pa[1]);
      tmpCtx.lineTo(pb[0], pb[1]);
      tmpCtx.stroke();
    }}
    var names = Object.keys(ckp);
    for (var j = 0; j < names.length; j++) {{
      var p = [ckp[names[j]][0]*w, ckp[names[j]][1]*h];
      tmpCtx.fillStyle = '#FFFFFF';
      tmpCtx.beginPath();
      tmpCtx.arc(p[0], p[1], 5, 0, Math.PI*2);
      tmpCtx.fill();
    }}

    var dataUrl = tmpCanvas.toDataURL('image/png');
    var hiddenEl = document.querySelector('#' + id + '_output textarea, #' + id + '_output input');
    if (hiddenEl) {{
      var nativeSet = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype, 'value'
      );
      if (!nativeSet) nativeSet = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
      if (nativeSet && nativeSet.set) {{
        nativeSet.set.call(hiddenEl, dataUrl);
      }} else {{
        hiddenEl.value = dataUrl;
      }}
      hiddenEl.dispatchEvent(new Event('input', {{bubbles: true}}));
      hiddenEl.dispatchEvent(new Event('change', {{bubbles: true}}));
    }}
  }};

  draw();
}})();
</script>
"""


def build_pose_editor(editor_id: str = 'freyra_pose_editor'):
    """Build the pose editor component.

    Returns (html_component, output_textbox).
    The output_textbox receives a base64 data URL of the stick figure when 'Apply Pose' is clicked.
    """
    html = _build_pose_editor_html(editor_id)
    html_component = gr.HTML(value=html)

    output_textbox = gr.Textbox(
        value='',
        visible=False,
        elem_id=f'{editor_id}_output',
        label='Pose Editor Output',
    )

    return html_component, output_textbox
