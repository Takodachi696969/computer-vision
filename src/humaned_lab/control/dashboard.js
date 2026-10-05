/* Browser controls use SI units at the API boundary; no physical hardware calls. */
"use strict";
const $ = id => document.getElementById(id);
const degrees = radians => radians * 180 / Math.PI;
const radians = angle => angle * Math.PI / 180;
const physicsLabels = {
  gravity_enabled: "Gravity",
  robot_object_contacts_enabled: "Arm ↔ objects",
  object_floor_contacts_enabled: "Objects ↔ floor / obstacles",
  object_object_contacts_enabled: "Objects ↔ objects",
  robot_floor_contacts_enabled: "Arm ↔ floor / obstacles",
  friction_enabled: "Contact friction",
  joint_damping_enabled: "Passive joint damping",
  actuation_enabled: "Position actuators"
};
const baseline = {kp: [180,180,140,50,45,30], kv: [10,10,8,3,3,2], torque_limits_nm: [300,300,300,300,300,300], target_velocity_limits_rad_s: null};
let state = null, mode = "joints", editingJoints = false, polling = false;
let selectedJoint = 0, keyboardEnabled = false, jogSign = 0, jogTimer = null, jogPending = false;
let settingsInitialized = false, cubeFingerprint = "", actuatorFingerprint = "", physicsFingerprint = "", viewerFingerprint = "";
let waypoints = [];

function notify(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}

async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : {"Content-Type": "application/json"},
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; } catch { detail = response.statusText; }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json();
}

function number(id) {
  const field = $(id);
  if (field.value.trim() === "" || !field.checkValidity()) throw new Error(`Check ${field.parentElement.textContent.trim()}.`);
  const value = Number(field.value);
  if (!Number.isFinite(value)) throw new Error("All values must be finite numbers.");
  return value;
}
const vector = ids => ids.map(number);
function setVector(ids, values) { ids.forEach((id, i) => $(id).value = values[i]); }

async function action(fn, message) {
  try {
    const result = await fn();
    if (result && result.q_rad) show(result);
    notify(message);
  } catch (error) { notify(error.message, true); }
}

function createControls(s) {
  $("joints").replaceChildren();
  s.q_rad.forEach((q, i) => {
    const row = document.createElement("div"); row.className = "joint";
    const label = document.createElement("label"); label.htmlFor = `j${i}`; label.append(`J${i+1}`);
    const value = document.createElement("span"); value.id = `val${i}`; label.append(value);
    const input = document.createElement("input"); input.type = "range"; input.id = `j${i}`;
    input.min = degrees(s.joint_limits_rad[i][0]); input.max = degrees(s.joint_limits_rad[i][1]); input.step = "any";
    input.value = degrees(q); value.textContent = `${degrees(q).toFixed(1)}°`;
    input.oninput = () => { editingJoints = true; value.textContent = `${Number(input.value).toFixed(1)}°`; };
    row.append(label, input); $("joints").append(row);
  });
  $("jointPicker").replaceChildren();
  s.q_rad.forEach((_, i) => {
    const button = document.createElement("button"); button.textContent = `J${i+1}`;
    button.onclick = () => selectJoint(i); $("jointPicker").append(button);
  });
  for (const [key, label] of Object.entries(physicsLabels)) {
    const node = document.createElement("label"); node.className = "check";
    const input = document.createElement("input"); input.type = "checkbox"; input.id = key;
    node.append(input, label); $("physicsChecks").append(node);
  }
  for (let i = 0; i < 6; i++) {
    const row = document.createElement("tr");
    const name = document.createElement("td"); name.textContent = `J${i+1}`; row.append(name);
    for (const prefix of ["cap", "kp", "kv"]) {
      const cell = document.createElement("td"); const input = document.createElement("input");
      input.type = "number"; input.id = `${prefix}${i}`; input.step = "any";
      input.min = prefix === "cap" ? "0.001" : "0"; input.setAttribute("aria-label", `J${i+1} ${prefix}`);
      cell.append(input); row.append(cell);
    }
    $("actuatorInputs").append(row);
  }
  selectJoint(0); settingsInitialized = true;
}

function fillCube(cube) {
  const props = cube.properties || cube;
  setVector(["sx","sy","sz"], props.size_m || [0.04,0.04,0.04]);
  $("mass").value = props.mass_kg ?? 0.08;
  setVector(["comx","comy","comz"], props.com_offset_m || [0,0,0]);
  setVector(["sliding","torsional","rolling"], props.friction || [0.8,0.02,0.002]);
  $("useRestitution").checked = props.restitution !== null && props.restitution !== undefined;
  $("restitution").value = props.restitution ?? 0.3;
}

function fillActuators(settings) {
  settings = {...baseline, ...settings};
  for (let i = 0; i < 6; i++) {
    $("cap"+i).value = settings.torque_limits_nm[i];
    $("kp"+i).value = settings.kp[i]; $("kv"+i).value = settings.kv[i];
  }
  $("limitTargetSpeed").checked = settings.target_velocity_limits_rad_s !== null;
  $("targetSpeed").value = settings.target_velocity_limits_rad_s?.[0] ?? 0.5;
}

function updateLoads(s) {
  const torques = s.joint_measured_torque_nm || Array(6).fill(0);
  const limits = s.actuator_settings?.torque_limits_nm || baseline.torque_limits_nm;
  const saturated = s.joint_torque_saturated || Array(6).fill(false);
  $("loads").replaceChildren();
  for (let i = 0; i < 6; i++) {
    const row = document.createElement("tr"); row.classList.toggle("saturated", saturated[i]);
    const ratio = Math.min(1, Math.abs(torques[i]) / limits[i]);
    const values = [`J${i+1}`, `${degrees(s.q_rad[i]).toFixed(1)} / ${degrees(s.joint_targets_rad[i]).toFixed(1)}`, torques[i].toFixed(2)];
    values.forEach(text => { const cell = document.createElement("td"); cell.textContent = text; row.append(cell); });
    const cell = document.createElement("td"), track = document.createElement("span"), bar = document.createElement("i");
    track.className = "load-track"; bar.style.width = `${ratio * 100}%`; track.append(bar);
    cell.append(track, `${(ratio * 100).toFixed(0)}%${saturated[i] ? " · CAP" : ""}`); row.append(cell); $("loads").append(row);
  }
}

function show(s) {
  state = s;
  if (!settingsInitialized) createControls(s);
  if (!editingJoints) s.joint_targets_rad.forEach((q, i) => { $("j"+i).value = degrees(q); $("val"+i).textContent = `${degrees(q).toFixed(1)}°`; });
  $("connection").textContent = s.physics_error ? "PHYSICS ERROR" : s.running ? "RUNNING" : "PAUSED";
  $("pause").textContent = s.running ? "Pause" : "Resume";
  $("clock").textContent = `${s.time_s.toFixed(2)} s`;
  $("tcp").textContent = s.tcp_m.map(x => x.toFixed(3)).join(" / ");
  const viewer = s.viewer || s.viewer_settings || {};
  const metrics = s.viewer_metrics || viewer;
  $("fps").textContent = Number(metrics.actual_fps ?? metrics.fps ?? s.render_fps ?? 0).toFixed(1);
  $("realtime").textContent = `${Number(metrics.physics_realtime_factor ?? metrics.realtime_factor ?? s.realtime_factor ?? 0).toFixed(2)}×`;
  $("renderTiming").textContent = ` Render + encode ${Number(metrics.render_ms ?? 0).toFixed(1)} ms`;
  const viewerConfig = [viewer.target_fps, viewer.width, viewer.height, viewer.jpeg_quality, viewer.show_collisions];
  if (viewer.target_fps && JSON.stringify(viewerConfig) !== viewerFingerprint) {
    $("targetFps").value = viewer.target_fps;
    $("resolution").value = `${viewer.width},${viewer.height}`;
    $("jpegQuality").value = viewer.jpeg_quality;
    $("showCollisions").checked = viewer.show_collisions;
    viewerFingerprint = JSON.stringify(viewerConfig);
  }
  if (s.render_error) notify(`Rendering unavailable: ${s.render_error}`, true);
  if (s.physics_error) notify(`Physics stopped: ${s.physics_error}`, true);
  const names = s.cubes.map(cube => cube.name);
  if (JSON.stringify([...$("cube").options].map(o => o.value)) !== JSON.stringify(names)) {
    const previous = $("cube").value; $("cube").replaceChildren();
    names.forEach(name => { const option = document.createElement("option"); option.value = name; option.textContent = name; $("cube").append(option); });
    if (names.includes(previous)) $("cube").value = previous;
  }
  const cube = s.cubes.find(c => c.name === $("cube").value);
  if (cube) {
    const props = cube.properties || Object.fromEntries(["size_m","mass_kg","com_offset_m","friction","restitution"].map(key => [key, cube[key]]));
    const fingerprint = JSON.stringify([cube.name, props]);
    if (fingerprint !== cubeFingerprint) { fillCube(cube); cubeFingerprint = fingerprint; }
    $("cubeState").textContent = `Position ${cube.position_m.map(v => v.toFixed(3)).join(", ")} m · contact ${Number(cube.contact_force_n ?? 0).toFixed(2)} N · arm contact ${Number(s.robot_contact_force_n ?? 0).toFixed(2)} N`;
  }
  $("applyCube").disabled = $("launchCube").disabled = !cube;
  const physics = s.physics_settings || {};
  if (JSON.stringify(physics) !== physicsFingerprint) {
    Object.keys(physicsLabels).forEach(key => $(key).checked = physics[key] ?? true);
    physicsFingerprint = JSON.stringify(physics);
  }
  const actuators = s.actuator_settings || baseline;
  if (JSON.stringify(actuators) !== actuatorFingerprint) { fillActuators(actuators); actuatorFingerprint = JSON.stringify(actuators); }
  updateLoads(s);
  const path = s.trajectory;
  if (path && path.active) {
    const index = path.waypoint_index ?? path.current_waypoint ?? 0;
    const count = path.waypoint_count ?? path.total_waypoints ?? "?";
    const error = path.error_m ?? path.tcp_error_m ?? path.distance_m;
    $("pathStatus").textContent = `Waypoint ${index+1}/${count} · ${path.stalled ? "STALLED / target not reached" : !s.running ? "Paused" : path.planned_complete ? "Settling at target" : "Moving"}${error == null ? "" : ` · tip error ${Number(error).toFixed(4)} m`}`;
  } else if (path) $("pathStatus").textContent = path.reached ? "Sequence reached its final target." : path.stalled ? "Stopped: target not reached under current dynamics." : path.status === "force_limit" ? "Stopped: arm contact-force limit exceeded." : path.status && path.status !== "idle" ? `Sequence status: ${path.status.replaceAll("_", " ")}` : "No active sequence.";
  if (jogSign && !s.running) stopJog();
}

function selectJoint(index) {
  if (jogSign) stopJog();
  selectedJoint = (index + 6) % 6;
  $("selectedJoint").textContent = `J${selectedJoint+1}`;
  [...$("jointPicker").children].forEach((button, i) => button.classList.toggle("active", i === selectedJoint));
}

async function heartbeat() {
  if (!jogSign || jogPending) return;
  jogPending = true;
  try {
    const sign = jogSign;
    const joint = selectedJoint;
    await api("/api/jog", {joint_index: joint, velocity_rad_s: sign * radians(number("jogRate"))});
    if (!jogSign || joint !== selectedJoint || sign !== jogSign) await api("/api/jog/stop", {});
  } catch (error) { notify(error.message, true); stopJog(); }
  finally { jogPending = false; }
}

function startJog(sign) {
  if (!state || !state.running) { notify("Resume physics before jogging.", true); return; }
  if (jogSign === sign) return;
  jogSign = sign;
  clearInterval(jogTimer); jogTimer = setInterval(heartbeat, 100); heartbeat();
  $("jogStatus").textContent = `Moving J${selectedJoint+1} ${sign > 0 ? "+" : "−"} at target rate ${$("jogRate").value}°/s`;
}

function stopJog(unloading = false) {
  const wasMoving = Boolean(jogSign); jogSign = 0; clearInterval(jogTimer); jogTimer = null;
  if (wasMoving) {
    fetch("/api/jog/stop", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}", keepalive:unloading}).catch(() => {});
  }
  $("jogStatus").textContent = keyboardEnabled ? `Arrow controls ready · J${selectedJoint+1}` : "Arrow controls disabled.";
}

function setMode(next) {
  stopJog(); keyboardEnabled = false; $("enableKeyboard").textContent = "Enable arrow controls";
  mode = next;
  for (const name of ["joints","tip","jog"]) {
    $("mode-"+name).hidden = name !== mode;
    $("tab-"+name).classList.toggle("active", name === mode);
    $("tab-"+name).setAttribute("aria-selected", String(name === mode));
  }
}

function drawWaypoints() {
  $("waypoints").replaceChildren();
  waypoints.forEach((point, index) => {
    const item = document.createElement("li"); item.textContent = point.map(v => v.toFixed(3)).join(", ")+" m";
    const remove = document.createElement("button"); remove.textContent = "×"; remove.setAttribute("aria-label", `Remove waypoint ${index+1}`);
    remove.onclick = () => { waypoints.splice(index,1); drawWaypoints(); }; item.append(remove); $("waypoints").append(item);
  });
}

function pathOptions() {
  return {tolerance_m:number("tipTolerance") / 1000,settle_timeout_s:number("settleTimeout"),force_limit_n:$("useContactStop").checked ? number("contactStop") : null};
}

document.querySelectorAll("[data-mode]").forEach(button => button.onclick = () => setMode(button.dataset.mode));
$("pause").onclick = () => action(async () => { stopJog(); return api("/api/running", {running:!state.running}); }, "Playback updated.");
$("reset").onclick = () => action(async () => { stopJog(); editingJoints = false; return api("/api/reset", {seed:0}); }, "World reset. Physical settings retained.");
$("applyJoints").onclick = () => action(async () => { stopJog(); const result = await api("/api/joints", {q_rad:Array.from({length:6},(_,i)=>radians(number("j"+i)))}); editingJoints = false; return result; }, "Joint targets applied.");
$("applyViewer").onclick = () => action(() => { const [width,height] = $("resolution").value.split(",").map(Number); return api("/api/viewer", {target_fps:number("targetFps"), width,height,jpeg_quality:number("jpegQuality"),show_collisions:$("showCollisions").checked}); }, "Viewer settings applied.");
$("useTip").onclick = () => state && setVector(["tx","ty","tz"], state.tcp_m.map(v => Number(v.toFixed(4))));
$("moveTip").onclick = () => action(async () => { stopJog(); return api("/api/tcp", {position_m:vector(["tx","ty","tz"]),duration_s:number("duration"),...pathOptions()}); }, "IK solved. Watch measured tip position and loads.");
$("appendWaypoint").onclick = () => { try { waypoints.push(vector(["tx","ty","tz"])); drawWaypoints(); } catch(error) { notify(error.message,true); } };
$("clearWaypoints").onclick = () => { waypoints=[]; drawWaypoints(); };
$("runWaypoints").onclick = () => action(async () => { if (!waypoints.length) throw new Error("Add at least one waypoint."); stopJog(); return api("/api/trajectory", {waypoints_m:waypoints,segment_duration_s:number("duration"),loop:$("loopPath").checked,...pathOptions()}); }, "Waypoint sequence started; actual tip arrival controls progression.");
$("stopWaypoints").onclick = () => action(() => api("/api/trajectory/stop", {}), "Sequence stopped. Holding measured joints.");
$("cube").onchange = () => { cubeFingerprint=""; if(state) show(state); };
$("applyCube").onclick = () => action(() => api("/api/cubes/"+encodeURIComponent($("cube").value)+"/properties", {size_m:vector(["sx","sy","sz"]),mass_kg:number("mass"),com_offset_m:vector(["comx","comy","comz"]),friction:vector(["sliding","torsional","rolling"]),restitution:$("useRestitution").checked ? number("restitution") : null}), "Cube properties applied. Check overlaps after changing dimensions.");
$("launchCube").onclick = () => action(() => api("/api/cubes/"+encodeURIComponent($("cube").value), {position_m:vector(["cx","cy","cz"]),velocity_m_s:vector(["vx","vy","vz"])}), "Cube placed. Resume physics to watch motion.");
$("applyPhysics").onclick = () => action(() => api("/api/physics", Object.fromEntries(Object.keys(physicsLabels).map(key => [key,$(key).checked]))), "Physics layers applied.");
$("baselineActuators").onclick = () => { fillActuators(baseline); notify("Baseline values loaded into the editor; click Apply actuator model."); };
$("loadActuators").onclick = () => { fillActuators({...baseline,torque_limits_nm:[12,12,8,3,3,2],target_velocity_limits_rad_s:Array(6).fill(0.5)}); notify("Illustrative load preset loaded; click Apply actuator model. Values are not calibrated PAROL6 specifications."); };
$("applyActuators").onclick = () => action(() => api("/api/actuators", {kp:Array.from({length:6},(_,i)=>number("kp"+i)),kv:Array.from({length:6},(_,i)=>number("kv"+i)),torque_limits_nm:Array.from({length:6},(_,i)=>number("cap"+i)),target_velocity_limits_rad_s:$("limitTargetSpeed").checked ? Array(6).fill(number("targetSpeed")) : null}), "Actuator model applied. Watch torque saturation and tracking error.");
$("saveScene").onclick = () => action(async () => {
  const scene = await api("/api/scene"), blob = new Blob([JSON.stringify(scene,null,2)+"\n"], {type:"application/json"});
  const url = URL.createObjectURL(blob), link = document.createElement("a"); link.href=url; link.download="physics-scene.json";
  document.body.append(link); link.click(); link.remove(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}, "Scene downloaded. Launch with --scene to reproduce the physical settings.");
$("enableKeyboard").onclick = () => { keyboardEnabled=!keyboardEnabled; if(!keyboardEnabled) stopJog(); $("enableKeyboard").textContent=keyboardEnabled ? "Disable arrow controls" : "Enable arrow controls"; $("jogStatus").textContent=keyboardEnabled ? `Arrow controls ready · J${selectedJoint+1}` : "Arrow controls disabled."; };
for (const [id,sign] of [["jogNegative",-1],["jogPositive",1]]) {
  $(id).onpointerdown = event => { event.preventDefault(); $(id).setPointerCapture(event.pointerId); startJog(sign); };
  $(id).onpointerup = () => stopJog(); $(id).onpointercancel = () => stopJog(); $(id).onlostpointercapture = () => stopJog();
}
document.addEventListener("keydown", event => {
  if (mode!=="jog" || !keyboardEnabled || event.altKey || event.ctrlKey || event.metaKey) return;
  if (["INPUT","SELECT","TEXTAREA"].includes(event.target.tagName) || event.target.isContentEditable) { stopJog(); return; }
  if (!["ArrowUp","ArrowDown","ArrowLeft","ArrowRight","1","2","3","4","5","6"].includes(event.key)) return;
  event.preventDefault();
  if (event.key==="ArrowUp") startJog(1); else if (event.key==="ArrowDown") startJog(-1);
  else if (!event.repeat) selectJoint(event.key==="ArrowLeft" ? selectedJoint-1 : event.key==="ArrowRight" ? selectedJoint+1 : Number(event.key)-1);
});
document.addEventListener("keyup", event => { if (event.key==="ArrowUp" || event.key==="ArrowDown") stopJog(); });
window.addEventListener("blur",()=>stopJog(true));
document.addEventListener("visibilitychange",()=>{ if(document.hidden) stopJog(true); });
document.addEventListener("focusin",event=>{ if(["INPUT","SELECT","TEXTAREA"].includes(event.target.tagName)) stopJog(); });
window.addEventListener("pagehide",()=>stopJog(true));

async function poll() {
  if (polling) return; polling=true;
  try { show(await api("/api/state")); }
  catch(error) { $("connection").textContent="OFFLINE"; notify(error.message,true); stopJog(); }
  finally { polling=false; }
}
$("frame").onerror = () => { notify("Stream disconnected. Reconnecting…",true); setTimeout(()=>$("frame").src="/api/stream.mjpg?t="+Date.now(),1500); };
poll(); setInterval(poll,250);
