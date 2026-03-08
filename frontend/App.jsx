import { useState, useEffect, useCallback, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend, PieChart, Pie, Cell
} from "recharts";

// ─── API CONFIG ────────────────────────────────────────────────────────────────
const API = "http://localhost:8000";

// ─── COLOUR SYSTEM ─────────────────────────────────────────────────────────────
const VEHICLE_COLORS = ["#00E5FF", "#FF6B35", "#7CFC00", "#FF3CAC", "#FFD700"];
const TRAFFIC_COLORS = { Low: "#7CFC00", Medium: "#FFD700", High: "#FF6B35", Very_High: "#FF3CAC" };
const WEATHER_COLORS = ["#00E5FF","#B0BEC5","#64B5F6","#1565C0","#CFD8DC","#FF3CAC","#E0E0E0"];

// ─── HELPERS ───────────────────────────────────────────────────────────────────
const fmt = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : "—");
const api = async (path, opts = {}) => {
  const r = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};

// ─── UNIT-SQUARE MAP (coords in [0,1]) ─────────────────────────────────────────
function RouteMap({ routes, depot, orders }) {
  const svgRef = useRef(null);
  const W = 520, H = 480, PAD = 30;

  const toSvg = (lat, lon) => ({
    x: PAD + lon * (W - 2 * PAD),
    y: H - PAD - lat * (H - 2 * PAD),
  });

  return (
    <div style={{ position: "relative", background: "#050B14", borderRadius: 12, overflow: "hidden", border: "1px solid #0D2137" }}>
      <svg width={W} height={H} style={{ display: "block" }}>
        {/* grid */}
        {[0,.25,.5,.75,1].map(v => (
          <g key={v}>
            <line x1={PAD + v*(W-2*PAD)} y1={PAD} x2={PAD + v*(W-2*PAD)} y2={H-PAD}
                  stroke="#0D2137" strokeWidth={1} />
            <line x1={PAD} y1={H-PAD-v*(H-2*PAD)} x2={W-PAD} y2={H-PAD-v*(H-2*PAD)}
                  stroke="#0D2137" strokeWidth={1} />
          </g>
        ))}

        {/* all orders (dim background) */}
        {(orders || []).map((o, i) => {
          const p = toSvg(o.pickup_lat, o.pickup_lon);
          const d = toSvg(o.drop_lat, o.drop_lon);
          return (
            <g key={i} opacity={0.18}>
              <circle cx={p.x} cy={p.y} r={2} fill="#4FC3F7" />
              <circle cx={d.x} cy={d.y} r={2} fill="#81C784" />
              <line x1={p.x} y1={p.y} x2={d.x} y2={d.y} stroke="#ffffff" strokeWidth={0.5} />
            </g>
          );
        })}

        {/* routes */}
        {(routes || []).map((vehicle, vi) => {
          const col = VEHICLE_COLORS[vi % VEHICLE_COLORS.length];
          const stops = vehicle.stops || [];
          const pts = stops.flatMap(s => [
            toSvg(s.pickup_lat, s.pickup_lon),
            toSvg(s.drop_lat,   s.drop_lon),
          ]);
          if (depot) pts.unshift(toSvg(depot.lat, depot.lon));

          return (
            <g key={vi}>
              {pts.slice(1).map((pt, pi) => (
                <line key={pi}
                  x1={pts[pi].x} y1={pts[pi].y}
                  x2={pt.x}      y2={pt.y}
                  stroke={col} strokeWidth={1.5} opacity={0.75}
                  strokeDasharray={pi === 0 ? "none" : "none"}
                />
              ))}
              {stops.map((s, si) => {
                const p = toSvg(s.pickup_lat, s.pickup_lon);
                const d = toSvg(s.drop_lat,   s.drop_lon);
                return (
                  <g key={si}>
                    <circle cx={p.x} cy={p.y} r={4} fill={col} opacity={0.9} />
                    <circle cx={d.x} cy={d.y} r={3} fill="#ffffff" opacity={0.6} stroke={col} strokeWidth={1} />
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* depot */}
        {depot && (() => {
          const dp = toSvg(depot.lat, depot.lon);
          return (
            <g>
              <circle cx={dp.x} cy={dp.y} r={8} fill="none" stroke="#FFD700" strokeWidth={2} />
              <circle cx={dp.x} cy={dp.y} r={4} fill="#FFD700" />
              <text x={dp.x + 12} y={dp.y + 4} fill="#FFD700" fontSize={10} fontFamily="monospace">DEPOT</text>
            </g>
          );
        })()}
      </svg>

      {/* legend */}
      <div style={{ position: "absolute", bottom: 10, left: 10, display: "flex", gap: 12, flexWrap: "wrap" }}>
        {(routes || []).filter(v => v.stops?.length).map((v, vi) => (
          <div key={vi} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: VEHICLE_COLORS[vi % VEHICLE_COLORS.length] }} />
            <span style={{ color: "#B0BEC5", fontSize: 11, fontFamily: "monospace" }}>V{vi+1} ({v.stops.length} stops)</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── METRIC CARD ───────────────────────────────────────────────────────────────
function MetricCard({ label, value, unit, sub, color = "#00E5FF", icon }) {
  return (
    <div style={{
      background: "linear-gradient(135deg, #070F1C 0%, #0A1628 100%)",
      border: `1px solid ${color}22`,
      borderRadius: 10, padding: "14px 18px",
      boxShadow: `0 0 20px ${color}08`,
      flex: 1, minWidth: 130,
    }}>
      <div style={{ color: "#546E7A", fontSize: 11, fontFamily: "monospace", letterSpacing: 1, marginBottom: 4 }}>
        {icon} {label}
      </div>
      <div style={{ color, fontSize: 26, fontWeight: 700, fontFamily: "'Courier New', monospace", lineHeight: 1 }}>
        {value}
        <span style={{ fontSize: 13, color: "#546E7A", marginLeft: 4 }}>{unit}</span>
      </div>
      {sub && <div style={{ color: "#37474F", fontSize: 10, marginTop: 3, fontFamily: "monospace" }}>{sub}</div>}
    </div>
  );
}

// ─── EVENT PANEL ──────────────────────────────────────────────────────────────
function EventPanel({ onEvent, disabled }) {
  const [type, setType]   = useState("NEW_ORDER");
  const [status, setStatus] = useState(null);

  const fire = async () => {
    try {
      let payload = {};
      if (type === "NEW_ORDER") {
        payload = {
          order: {
            order_id: Math.floor(Math.random() * 99999),
            pickup_lat: Math.random() * 0.4 + 0.6,
            pickup_lon: Math.random() * 0.3 + 0.7,
            drop_lat:   Math.random() * 0.4 + 0.5,
            drop_lon:   Math.random() * 0.3 + 0.7,
            Road_traffic_density: Math.floor(Math.random() * 4),
            Weather_conditions:   Math.floor(Math.random() * 7),
            order_time_min: 800 + Math.floor(Math.random() * 400),
            pickup_time_min: 810 + Math.floor(Math.random() * 400),
          }
        };
      }
      setStatus("firing…");
      const res = await onEvent(type, payload);
      setStatus(`✅ ${res.strategy}  (${res.solve_time_s}s)`);
    } catch (e) {
      setStatus(`❌ ${e.message}`);
    }
  };

  const btnStyle = (active, col) => ({
    padding: "6px 14px", borderRadius: 6, cursor: "pointer", fontSize: 12,
    fontFamily: "monospace", border: `1px solid ${active ? col : "#1A2744"}`,
    background: active ? `${col}22` : "transparent",
    color: active ? col : "#546E7A", transition: "all .15s",
  });

  return (
    <div style={{ background: "#070F1C", border: "1px solid #0D2137", borderRadius: 10, padding: 16 }}>
      <div style={{ color: "#B0BEC5", fontSize: 12, fontFamily: "monospace", marginBottom: 10, letterSpacing: 1 }}>
        ⚡ REAL-TIME EVENTS
      </div>
      <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
        {["NEW_ORDER","TRAFFIC_UPDATE","DELAY"].map(t => (
          <button key={t} onClick={() => setType(t)} style={btnStyle(type===t, "#00E5FF")}>
            {t === "NEW_ORDER" ? "📦" : t === "TRAFFIC_UPDATE" ? "🚦" : "⏱️"} {t}
          </button>
        ))}
      </div>
      <button onClick={fire} disabled={disabled}
        style={{
          width: "100%", padding: "8px 0", borderRadius: 8, cursor: disabled ? "not-allowed" : "pointer",
          background: disabled ? "#0D2137" : "linear-gradient(90deg,#00E5FF22,#00E5FF44)",
          border: "1px solid #00E5FF44", color: disabled ? "#37474F" : "#00E5FF",
          fontFamily: "monospace", fontSize: 13, letterSpacing: 1,
        }}>
        FIRE EVENT
      </button>
      {status && (
        <div style={{ marginTop: 8, color: "#7CFC00", fontSize: 11, fontFamily: "monospace" }}>{status}</div>
      )}
    </div>
  );
}

// ─── ROUTE TABLE ───────────────────────────────────────────────────────────────
function RouteTable({ routes }) {
  const [open, setOpen] = useState(null);
  if (!routes?.length) return null;
  return (
    <div>
      {routes.filter(v => v.stops?.length).map((v, vi) => (
        <div key={vi} style={{ marginBottom: 6 }}>
          <div onClick={() => setOpen(open === vi ? null : vi)}
            style={{
              display: "flex", alignItems: "center", gap: 10, padding: "8px 14px",
              background: "#070F1C", borderRadius: 8, cursor: "pointer",
              border: `1px solid ${VEHICLE_COLORS[vi%VEHICLE_COLORS.length]}33`,
            }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: VEHICLE_COLORS[vi%VEHICLE_COLORS.length] }} />
            <span style={{ color: "#B0BEC5", fontFamily: "monospace", fontSize: 13 }}>
              Vehicle {vi+1}
            </span>
            <span style={{ marginLeft: "auto", color: "#546E7A", fontSize: 11 }}>
              {v.stops.length} stops  {open === vi ? "▲" : "▼"}
            </span>
          </div>
          {open === vi && (
            <div style={{ background: "#050B14", borderRadius: "0 0 8px 8px", border: "1px solid #0D2137", borderTop: "none" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "monospace" }}>
                <thead>
                  <tr style={{ background: "#0A1628" }}>
                    {["#","ORDER ID","DIST (km)","ETA (min)","TRAFFIC","WEATHER"].map(h => (
                      <th key={h} style={{ padding: "6px 10px", color: "#546E7A", textAlign: "left", fontWeight: 400 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {v.stops.map((s, si) => (
                    <tr key={si} style={{ borderTop: "1px solid #0D2137" }}>
                      <td style={{ padding: "5px 10px", color: "#37474F" }}>{si + 1}</td>
                      <td style={{ color: "#B0BEC5" }}>{s.order_id ?? "—"}</td>
                      <td style={{ color: "#00E5FF" }}>{fmt(s.distance_km)}</td>
                      <td style={{ color: "#FFD700" }}>{fmt(s.eta_min, 1)}</td>
                      <td style={{ color: TRAFFIC_COLORS[s.traffic] || "#B0BEC5" }}>
                        {s.traffic}
                      </td>
                      <td style={{ color: "#78909C" }}>{s.weather}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ─── MAIN APP ──────────────────────────────────────────────────────────────────
export default function App() {
  const [orders,      setOrders]      = useState([]);
  const [routes,      setRoutes]      = useState([]);
  const [metrics,     setMetrics]     = useState(null);
  const [stats,       setStats]       = useState(null);
  const [rewardCurve, setRewardCurve] = useState(null);
  const [depot,       setDepot]       = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [tab,         setTab]         = useState("map");
  const [nOrders,     setNOrders]     = useState(25);
  const [nVehicles,   setNVehicles]   = useState(3);
  const [capacity,    setCapacity]    = useState(10);
  const [solveInfo,   setSolveInfo]   = useState(null);
  const [log,         setLog]         = useState([]);
  const [apiOk,       setApiOk]       = useState(null);

  const addLog = (msg) => setLog(l => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...l.slice(0,19)]);

  // check backend on mount
  useEffect(() => {
    api("/health").then(() => setApiOk(true)).catch(() => setApiOk(false));
    api("/dataset-stats").then(s => setStats(s)).catch(() => {});
    api("/reward-curve").then(d => setRewardCurve(d)).catch(() => {});
  }, []);

  const loadOrders = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api(`/orders?limit=${nOrders}`);
      setOrders(res.orders);
      addLog(`Loaded ${res.orders.length} orders from dataset`);
    } catch (e) {
      addLog(`❌ Load orders failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [nOrders]);

  const runOptimize = async () => {
    if (!orders.length) { addLog("Load orders first"); return; }
    setLoading(true);
    try {
      const res = await api("/optimize", {
        method: "POST",
        body: JSON.stringify({ orders, n_vehicles: nVehicles, capacity, time_limit: 10 }),
      });
      setRoutes(res.routes_detail || []);
      setMetrics(res.metrics);
      setSolveInfo({ status: res.status, time: res.solve_time_s, dist: res.total_dist_km });
      // compute depot from orders
      const lat = orders.reduce((s, o) => s + o.pickup_lat, 0) / orders.length;
      const lon = orders.reduce((s, o) => s + o.pickup_lon, 0) / orders.length;
      setDepot({ lat, lon });
      addLog(`✅ Optimised: ${res.status}  dist=${res.total_dist_km} km  solve=${res.solve_time_s}s`);
    } catch (e) {
      addLog(`❌ Optimize failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleEvent = async (type, payload) => {
    const res = await api("/event", {
      method: "POST",
      body: JSON.stringify({ event_type: type, payload }),
    });
    setRoutes(res.routes_detail || []);
    setMetrics(res.metrics);
    addLog(`Event ${type} → ${res.strategy}  dist=${res.total_dist_km} km`);
    return res;
  };

  // ── styles ──────────────────────────────────────────────────────────────────
  const base = {
    minHeight: "100vh",
    background: "#030912",
    color: "#B0BEC5",
    fontFamily: "'Courier New', monospace",
    padding: "0 0 40px",
  };

  const tabBtn = (t) => ({
    padding: "7px 18px", cursor: "pointer", fontSize: 12, letterSpacing: 1,
    border: "none", borderBottom: tab === t ? "2px solid #00E5FF" : "2px solid transparent",
    background: "transparent", color: tab === t ? "#00E5FF" : "#546E7A",
    fontFamily: "'Courier New', monospace", transition: "all .15s",
  });

  const ctrlBtn = (col = "#00E5FF", disabled = false) => ({
    padding: "9px 22px", borderRadius: 8, border: `1px solid ${col}55`,
    background: disabled ? "#0A1628" : `${col}18`,
    color: disabled ? "#37474F" : col,
    cursor: disabled ? "not-allowed" : "pointer",
    fontFamily: "monospace", fontSize: 13, letterSpacing: 1,
    transition: "all .15s",
  });

  const inputStyle = {
    background: "#070F1C", border: "1px solid #0D2137", borderRadius: 6,
    color: "#00E5FF", fontFamily: "monospace", fontSize: 13,
    padding: "6px 10px", width: 60, textAlign: "center",
  };

  // ── reward chart data ────────────────────────────────────────────────────────
  const rewardData = rewardCurve
    ? rewardCurve.steps.map((s, i) => ({ step: s / 1000, reward: rewardCurve.rewards[i] }))
    : [];

  // ── traffic/weather chart data ───────────────────────────────────────────────
  const trafficData = stats
    ? Object.entries(stats.traffic_dist).map(([k, v]) => ({ name: k, orders: v }))
    : [];

  const weatherData = stats
    ? Object.entries(stats.weather_dist).map(([k, v]) => ({ name: k, value: v }))
    : [];

  return (
    <div style={base}>
      {/* ── HEADER ─────────────────────────────────────────────────────────── */}
      <div style={{
        background: "linear-gradient(180deg, #050D1A 0%, #030912 100%)",
        borderBottom: "1px solid #0D2137", padding: "18px 32px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div>
          <div style={{ color: "#00E5FF", fontSize: 18, fontWeight: 700, letterSpacing: 2 }}>
            ◈ ROUTE OPTIMISER
          </div>
          <div style={{ color: "#37474F", fontSize: 10, letterSpacing: 3, marginTop: 2 }}>
            HYBRID DRL + OR-TOOLS · REAL-TIME DELIVERY
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {apiOk !== null && (
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{
                width: 8, height: 8, borderRadius: "50%",
                background: apiOk ? "#7CFC00" : "#FF3CAC",
                boxShadow: `0 0 8px ${apiOk ? "#7CFC00" : "#FF3CAC"}`,
              }} />
              <span style={{ color: "#546E7A", fontSize: 11 }}>
                {apiOk ? "API ONLINE" : "API OFFLINE"}
              </span>
            </div>
          )}
          {solveInfo && (
            <div style={{ color: "#546E7A", fontSize: 11 }}>
              {solveInfo.status} · {solveInfo.time}s
            </div>
          )}
        </div>
      </div>

      {/* ── OFFLINE WARNING ─────────────────────────────────────────────────── */}
      {apiOk === false && (
        <div style={{
          margin: "16px 32px", padding: "12px 18px", borderRadius: 8,
          background: "#FF3CAC11", border: "1px solid #FF3CAC33",
          color: "#FF3CAC", fontSize: 12,
        }}>
          ⚠ Backend is offline. Start the server: <code style={{ background: "#0D0D0D", padding: "2px 6px", borderRadius: 4 }}>
            cd backend && uvicorn main:app --reload
          </code>
          &nbsp;— The UI is fully rendered; all data requires the API.
        </div>
      )}

      {/* ── CONTROLS ───────────────────────────────────────────────────────── */}
      <div style={{ padding: "16px 32px", display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", borderBottom: "1px solid #0D2137" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: "#546E7A", fontSize: 11 }}>ORDERS</span>
          <input type="number" value={nOrders} min={5} max={200}
            onChange={e => setNOrders(+e.target.value)} style={inputStyle} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: "#546E7A", fontSize: 11 }}>VEHICLES</span>
          <input type="number" value={nVehicles} min={1} max={10}
            onChange={e => setNVehicles(+e.target.value)} style={inputStyle} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: "#546E7A", fontSize: 11 }}>CAPACITY</span>
          <input type="number" value={capacity} min={1} max={50}
            onChange={e => setCapacity(+e.target.value)} style={inputStyle} />
        </div>
        <button onClick={loadOrders} disabled={loading} style={ctrlBtn("#00E5FF", loading)}>
          {loading ? "⏳ LOADING…" : "📂 LOAD ORDERS"}
        </button>
        <button onClick={runOptimize} disabled={loading || !orders.length} style={ctrlBtn("#7CFC00", loading || !orders.length)}>
          ▶ OPTIMISE
        </button>
        {orders.length > 0 && (
          <span style={{ color: "#37474F", fontSize: 11 }}>{orders.length} orders ready</span>
        )}
      </div>

      {/* ── METRIC STRIP ───────────────────────────────────────────────────── */}
      {metrics && (
        <div style={{ padding: "14px 32px", display: "flex", gap: 12, flexWrap: "wrap" }}>
          <MetricCard label="TOTAL DISTANCE"   value={fmt(metrics.total_dist_km)}    unit="km"    color="#00E5FF" icon="📍" />
          <MetricCard label="TOTAL TIME"        value={fmt(metrics.total_time_min,0)} unit="min"   color="#FFD700" icon="⏱" />
          <MetricCard label="FUEL ESTIMATE"     value={fmt(metrics.total_fuel_L)}     unit="L"     color="#FF6B35" icon="⛽" />
          <MetricCard label="ORDERS SERVED"     value={metrics.orders_served}         unit=""      color="#7CFC00" icon="📦" />
          <MetricCard label="ON-TIME %"         value={metrics.on_time_pct}           unit="%"     color="#FF3CAC" icon="✅" />
          <MetricCard label="VEHICLES USED"     value={metrics.vehicles_used}         unit=""      color="#B39DDB" icon="🚚" />
        </div>
      )}

      {/* ── TABS ───────────────────────────────────────────────────────────── */}
      <div style={{ padding: "0 32px", borderBottom: "1px solid #0D2137", display: "flex", gap: 0 }}>
        {["map","routes","analytics","dataset","log"].map(t => (
          <button key={t} onClick={() => setTab(t)} style={tabBtn(t)}>
            {t.toUpperCase()}
          </button>
        ))}
      </div>

      {/* ── TAB: MAP ─────────────────────────────────────────────────────────*/}
      {tab === "map" && (
        <div style={{ padding: "20px 32px", display: "grid", gridTemplateColumns: "1fr 320px", gap: 20 }}>
          <RouteMap routes={routes} depot={depot} orders={orders} />
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <EventPanel onEvent={handleEvent} disabled={!routes.length} />
            {/* comparison table */}
            <div style={{ background: "#070F1C", border: "1px solid #0D2137", borderRadius: 10, padding: 16 }}>
              <div style={{ color: "#B0BEC5", fontSize: 12, letterSpacing: 1, marginBottom: 10 }}>
                📊 COMPARISON (Paper)
              </div>
              <table style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ color: "#546E7A" }}>
                    <th style={{ textAlign:"left", padding:"4px 6px" }}>Metric</th>
                    <th style={{ textAlign:"right" }}>OR-Tools</th>
                    <th style={{ textAlign:"right" }}>Hybrid</th>
                    <th style={{ textAlign:"right", color:"#7CFC00" }}>Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Distance","43.3 km","40.8 km","↓ 5.8%"],
                    ["Time","104 min","98 min","↓ 5.8%"],
                    ["Fuel","5.2 L","4.9 L","↓ 5.8%"],
                    ["Solve","12 s","5 s","↓ 58%"],
                    ["On-time","92%","94%","↑ 2%"],
                  ].map(([m,a,b,d]) => (
                    <tr key={m} style={{ borderTop:"1px solid #0D2137" }}>
                      <td style={{ padding:"5px 6px", color:"#78909C" }}>{m}</td>
                      <td style={{ textAlign:"right", color:"#546E7A" }}>{a}</td>
                      <td style={{ textAlign:"right", color:"#00E5FF" }}>{b}</td>
                      <td style={{ textAlign:"right", color:"#7CFC00", fontWeight:700 }}>{d}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB: ROUTES ───────────────────────────────────────────────────── */}
      {tab === "routes" && (
        <div style={{ padding: "20px 32px" }}>
          {routes.length === 0
            ? <div style={{ color: "#37474F", fontSize: 13 }}>Run optimisation to see routes.</div>
            : <RouteTable routes={routes} />}
        </div>
      )}

      {/* ── TAB: ANALYTICS ────────────────────────────────────────────────── */}
      {tab === "analytics" && (
        <div style={{ padding: "20px 32px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          {/* reward curve */}
          <div style={{ background: "#070F1C", border: "1px solid #0D2137", borderRadius: 10, padding: 16 }}>
            <div style={{ color: "#B0BEC5", fontSize: 12, letterSpacing: 1, marginBottom: 12 }}>
              📈 RL REWARD CURVE {rewardCurve?.source === "trained" ? "(TRAINED)" : "(REPRESENTATIVE)"}
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rewardData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#0D2137" />
                <XAxis dataKey="step" stroke="#546E7A" fontSize={10} label={{ value:"Steps (k)", fill:"#546E7A", position:"insideBottom", offset:-2, fontSize:10 }} />
                <YAxis stroke="#546E7A" fontSize={10} />
                <Tooltip contentStyle={{ background:"#050B14", border:"1px solid #0D2137", color:"#B0BEC5", fontSize:11 }} />
                <Line type="monotone" dataKey="reward" stroke="#00E5FF" strokeWidth={2} dot={{ r:3, fill:"#00E5FF" }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* traffic distribution */}
          <div style={{ background: "#070F1C", border: "1px solid #0D2137", borderRadius: 10, padding: 16 }}>
            <div style={{ color: "#B0BEC5", fontSize: 12, letterSpacing: 1, marginBottom: 12 }}>
              🚦 TRAFFIC DISTRIBUTION
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={trafficData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#0D2137" />
                <XAxis dataKey="name" stroke="#546E7A" fontSize={10} />
                <YAxis stroke="#546E7A" fontSize={10} />
                <Tooltip contentStyle={{ background:"#050B14", border:"1px solid #0D2137", color:"#B0BEC5", fontSize:11 }} />
                <Bar dataKey="orders" radius={[4,4,0,0]}>
                  {trafficData.map((e,i) => (
                    <Cell key={i} fill={TRAFFIC_COLORS[e.name] || "#546E7A"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* weather distribution */}
          <div style={{ background: "#070F1C", border: "1px solid #0D2137", borderRadius: 10, padding: 16 }}>
            <div style={{ color: "#B0BEC5", fontSize: 12, letterSpacing: 1, marginBottom: 12 }}>
              🌦 WEATHER DISTRIBUTION
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={weatherData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`} labelLine={{ stroke:"#0D2137" }}>
                  {weatherData.map((_,i) => <Cell key={i} fill={WEATHER_COLORS[i % WEATHER_COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background:"#050B14", border:"1px solid #0D2137", color:"#B0BEC5", fontSize:11 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* dynamic re-opt table */}
          <div style={{ background: "#070F1C", border: "1px solid #0D2137", borderRadius: 10, padding: 16 }}>
            <div style={{ color: "#B0BEC5", fontSize: 12, letterSpacing: 1, marginBottom: 12 }}>
              ⚡ DYNAMIC RE-OPT (Paper Results)
            </div>
            <table style={{ width:"100%", fontSize:12, borderCollapse:"collapse" }}>
              <thead>
                <tr style={{ color:"#546E7A" }}>
                  <th style={{ textAlign:"left", padding:"5px 8px" }}>Scenario</th>
                  <th>Distance</th><th>Time</th><th>Fuel</th>
                </tr>
              </thead>
              <tbody>
                {[["Before","43.3 km","104 min","5.2 L"],["After (new order)","45.0 km","108 min","5.4 L"]].map(([s,...v]) => (
                  <tr key={s} style={{ borderTop:"1px solid #0D2137" }}>
                    <td style={{ padding:"6px 8px", color: s.startsWith("After") ? "#FF6B35" : "#B0BEC5" }}>{s}</td>
                    {v.map((x,i) => <td key={i} style={{ textAlign:"center", color:"#78909C" }}>{x}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ marginTop:10, color:"#37474F", fontSize:11 }}>
              Observation: slight cost increase — system maintains feasibility
            </div>
          </div>
        </div>
      )}

      {/* ── TAB: DATASET ──────────────────────────────────────────────────── */}
      {tab === "dataset" && (
        <div style={{ padding: "20px 32px" }}>
          {stats ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 24 }}>
              <MetricCard label="TOTAL ORDERS"    value={stats.total_orders.toLocaleString()} unit=""     color="#00E5FF" icon="📦" />
              <MetricCard label="AVG DISTANCE"    value={fmt(stats.avg_distance_km)}           unit="km"   color="#FFD700" icon="📍" />
              <MetricCard label="AVG TIME TAKEN"  value={fmt(stats.avg_time_taken_min,1)}       unit="min"  color="#FF6B35" icon="⏱" />
              <MetricCard label="AVG WAIT TIME"   value={fmt(stats.avg_wait_time_min,1)}        unit="min"  color="#7CFC00" icon="⌛" />
              <MetricCard label="AVG FUEL"        value={fmt(stats.avg_fuel_L,3)}               unit="L"    color="#B39DDB" icon="⛽" />
              <MetricCard label="DEPOT"
                value={`${fmt(stats.depot.lat,3)}, ${fmt(stats.depot.lon,3)}`}
                unit="" color="#FF3CAC" icon="🏭"
                sub="centroid of all pickups"
              />
            </div>
          ) : <div style={{ color:"#37474F" }}>Loading dataset stats…</div>}

          {/* preprocessing steps */}
          <div style={{ background:"#070F1C", border:"1px solid #0D2137", borderRadius:10, padding:18 }}>
            <div style={{ color:"#B0BEC5", fontSize:12, letterSpacing:1, marginBottom:12 }}>🧹 PREPROCESSING PIPELINE</div>
            {[
              ["1", "Load 45,584 raw rows",                               "#00E5FF"],
              ["2", "Remove 31 exact duplicate rows",                     "#7CFC00"],
              ["3", "Fix 6,260 timing inversions (pickup < order time)",  "#FFD700"],
              ["4", "Remove 5,560 extreme wait-time outliers (>120 min)","#FF6B35"],
              ["5", "IQR winsorise distance_km & time_taken_min",        "#B39DDB"],
              ["6", "Derive traffic-adjusted ETA from Haversine + speed","#00E5FF"],
              ["7", "Add fuel estimate (0.12 L/km)",                      "#7CFC00"],
              ["8", "Compute virtual depot (centroid)",                   "#FFD700"],
              ["9", "Add normalised time-of-day features",                "#FF3CAC"],
              ["✅","Final clean dataset: 39,993 rows × 22 columns",      "#7CFC00"],
            ].map(([n, desc, col]) => (
              <div key={n} style={{ display:"flex", gap:12, padding:"5px 0", borderBottom:"1px solid #0D2137" }}>
                <div style={{ color:col, width:20, fontWeight:700, flexShrink:0 }}>{n}</div>
                <div style={{ color:"#78909C", fontSize:12 }}>{desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── TAB: LOG ──────────────────────────────────────────────────────── */}
      {tab === "log" && (
        <div style={{ padding:"20px 32px" }}>
          <div style={{ background:"#050B14", border:"1px solid #0D2137", borderRadius:10, padding:16, maxHeight:400, overflowY:"auto" }}>
            {log.length === 0
              ? <div style={{ color:"#37474F" }}>No events yet.</div>
              : log.map((l, i) => (
                <div key={i} style={{ color: l.includes("❌") ? "#FF3CAC" : l.includes("✅") ? "#7CFC00" : "#546E7A", fontSize:11, padding:"3px 0", borderBottom:"1px solid #0D1A2A" }}>
                  {l}
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
