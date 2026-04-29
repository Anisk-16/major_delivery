import { useState, useEffect, useRef, useCallback } from "react";
import { TopBar } from "./components/TopBar";
import { ControlStrip } from "./components/ControlStrip";
import { MetricsBand } from "./components/MetricsBand";
import { MapTab } from "./components/MapTab";
import { RouteTable } from "./components/RouteTable";
import { AnalyticsTab } from "./components/AnalyticsTab";
import { ModelPerformanceTab } from "./components/ModelPerformanceTab";

const API = "http://localhost:8000";

const apiFetch = async (path, opts={}) => {
  const r = await fetch(API+path, { headers:{"Content-Type":"application/json"}, ...opts });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};

export default function App() {
  const [orders, setOrders] = useState([]);
  const [routes, setRoutes] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [stats, setStats] = useState(null);
  const [rewardData, setRewardData] = useState([]);
  const [trafficData, setTrafficData] = useState([]);
  const [weatherData, setWeatherData] = useState([]);
  const [modelScores, setModelScores] = useState(null);
  const [rlModelMetrics, setRlModelMetrics] = useState(null);
  const [depot, setDepot] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState("map");
  
  const [nOrders, setNOrders] = useState(20);
  const [nVehicles, setNVehicles] = useState(3);
  const [capacity, setCapacity] = useState(10);
  const [solveInfo, setSolveInfo] = useState(null);
  
  const [log, setLog] = useState([]);
  const [apiOk, setApiOk] = useState(null);
  const [leafletOk, setLeafletOk] = useState(false);
  const [selVehicle, setSelVehicle] = useState(null);
  
  const [wsStatus, setWsStatus] = useState("disconnected");
  const [schedRunning, setSchedRunning] = useState(false);
  const [schedInterval, setSchedInterval] = useState(300);
  const wsRef = useRef(null);

  const addLog = useCallback(msg => setLog(l => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...l.slice(0, 29)]), []);

  // Fetch initial health and stats
  useEffect(() => {
    apiFetch("/health").then(() => { setApiOk(true); addLog("✅ Backend connected"); }).catch(() => { setApiOk(false); addLog("❌ Backend offline"); });
    apiFetch("/dataset-stats").then(s => {
      setStats(s);
      setTrafficData(Object.entries(s.traffic_dist).map(([k,v]) => ({name:k, orders:v})));
      setWeatherData(Object.entries(s.weather_dist).map(([k,v]) => ({name:k, value:v})));
    }).catch(()=>{});
    apiFetch("/reward-curve").then(d => {
      setRewardData(d.steps.map((s,i) => ({step: s/1000, reward: d.rewards[i]})));
    }).catch(()=>{});
  }, [addLog]);

  // WebSocket Setup
  useEffect(() => {
    function connectWS() {
      const ws = new WebSocket("ws://localhost:8000/ws");
      wsRef.current = ws;
      ws.onopen = () => { setWsStatus("connected"); addLog("✅ WebSocket connected — live updates active"); };
      ws.onmessage = (e) => {
        try { 
          const msg = JSON.parse(e.data);
          handleWsMessage(msg); 
        } catch {}
      };
      ws.onclose = () => { setWsStatus("disconnected"); setTimeout(connectWS, 5000); };
      ws.onerror = () => { setWsStatus("error"); ws.close(); };
    }
    connectWS();
    return () => { if(wsRef.current) wsRef.current.close(); };
  }, [addLog]);

  function handleWsMessage(msg) {
    if (msg.routes_detail?.length) setRoutes(msg.routes_detail);
    if (msg.metrics) setMetrics(m => ({ ...m, ...msg.metrics, solve_time: msg.solve_time_s }));
    if (msg.model_scores) setModelScores(msg.model_scores);
    if (msg.rl_model_metrics) setRlModelMetrics(msg.rl_model_metrics);
    if (msg.type === "AUTO_REOPT") {
      addLog(`🔄 Auto re-opt → ${msg.strategy} · ${msg.solve_time_s}s`);
    } else if (msg.type === "EVENT_RESULT") {
      addLog(`⚡ Live Event Pushed → Routes Updated (${msg.solve_time_s}s)`);
    }
  }

  const loadOrders = async () => {
    setLoading(true);
    try {
      const res = await apiFetch(`/orders?limit=${nOrders}`);
      setOrders(res.orders);
      setModelScores(null);
      setRlModelMetrics(null);
      addLog(`✅ Loaded ${res.orders.length} orders`);
    } catch(e) { addLog(`❌ ${e.message}`); }
    finally { setLoading(false); }
  };

  const runOptimize = async () => {
    if (!orders.length) return;
    setLoading(true); setSolveInfo(null); setModelScores(null); setRlModelMetrics(null);
    try {
      const res = await apiFetch("/optimize", {
        method:"POST", 
        body: JSON.stringify({orders, n_vehicles: nVehicles, capacity, time_limit:10})
      });
      setRoutes(res.routes_detail || []);
      setMetrics({ ...res.metrics, solve_time: res.solve_time_s });
      setModelScores(res.model_scores || null);
      setRlModelMetrics(res.rl_model_metrics || null);
      setSolveInfo({
        status: res.status,
        time: res.solve_time_s,
        dist: res.total_dist_km,
        warmStart: res.warm_start,
      });
      
      const lat = orders.reduce((s,o) => s + o.pickup_lat, 0) / orders.length;
      const lon = orders.reduce((s,o) => s + o.pickup_lon, 0) / orders.length;
      setDepot({ lat, lon });
      
      addLog(`✅ Optimised — ${res.status} · ${res.total_dist_km} km · ${res.solve_time_s}s`);
    } catch(e) { addLog(`❌ ${e.message}`); }
    finally { setLoading(false); }
  };

  const handleEvent = async (type) => {
    let payload = {};
    if (type === "NEW_ORDER") {
      payload = { order:{
        order_id: Math.floor(Math.random()*99999),
        pickup_lat: 0.6 + Math.random()*0.4, pickup_lon: 0.7 + Math.random()*0.3,
        drop_lat: 0.5 + Math.random()*0.45,  drop_lon: 0.7 + Math.random()*0.3,
        Road_traffic_density: Math.floor(Math.random()*4),
        Weather_conditions: Math.floor(Math.random()*7),
        order_time_min: 800, pickup_time_min: 815,
      }};
    }
    const res = await apiFetch("/event", { method: "POST", body: JSON.stringify({ event_type: type, payload }) });
    setRoutes(res.routes_detail || []);
    setMetrics(m => ({ ...m, ...res.metrics }));
    setModelScores(res.model_scores || null);
    setRlModelMetrics(res.rl_model_metrics || null);
    addLog(`⚡ ${type} triggered → Strategy: ${res.strategy} inside ${res.solve_time_s}s`);
  };

  return (
    <>
      <TopBar 
        apiOk={apiOk} 
        leafletOk={leafletOk} 
        wsStatus={wsStatus} 
        schedRunning={schedRunning} 
        schedInterval={schedInterval} 
        solveInfo={solveInfo} 
      />
      
      {apiOk === false && (
        <div style={{ margin: "24px 32px 0", padding: "12px 16px", borderRadius: "8px", background: "rgba(255, 60, 172, 0.1)", border: "1px solid var(--brand-coral)", color: "var(--brand-coral)", fontSize: "13px" }} className="mono-text">
          ⚠ Backend is offline. Please run the FastAPI server to use the dashboard.
        </div>
      )}

      <ControlStrip 
        orders={orders} nOrders={nOrders} setNOrders={setNOrders} 
        nVehicles={nVehicles} setNVehicles={setNVehicles} 
        capacity={capacity} setCapacity={setCapacity}
        loading={loading} loadOrders={loadOrders} runOptimize={runOptimize}
      />

      <MetricsBand metrics={metrics} />

      <div style={{ padding: "0 32px", borderBottom: "1px solid var(--border-light)", display: "flex", gap: "24px", marginBottom: "24px" }}>
        {[["map", "GLOBAL MAP"], ["routes", "ACTIVE ROUTES"], ["analytics", "ML ANALYTICS"], ["performance", "MODEL PERFORMANCE"]].map(([t, l]) => (
          <button 
            key={t} 
            onClick={() => setTab(t)} 
            className="mono-text animate-fade-in"
            style={{ 
              padding: "12px 0", borderTop: "none", borderLeft: "none", borderRight: "none", 
              borderBottom: `2px solid ${tab === t ? "var(--brand-cyan)" : "transparent"}`,
              background: "transparent", color: tab === t ? "var(--brand-cyan)" : "var(--text-muted)", 
              cursor: "pointer", fontSize: "14px", fontWeight: tab === t ? 600 : 400, letterSpacing: "1px",
              transition: "all 0.2s"
            }}
          >
            {l}
          </button>
        ))}
      </div>

      {tab === "map" && (
        <MapTab 
          routes={routes} orders={orders} depot={depot} 
          selectedVehicle={selVehicle} setLeafletOk={setLeafletOk}
          onEvent={handleEvent} disabled={!routes.length} log={log} 
        />
      )}

      {tab === "routes" && (
        <RouteTable routes={routes} />
      )}

      {tab === "analytics" && (
        <AnalyticsTab
          rewardData={rewardData}
          trafficData={trafficData}
          weatherData={weatherData}
          metrics={metrics}
          solveInfo={solveInfo}
        />
      )}

      {tab === "performance" && (
        <ModelPerformanceTab rlMetrics={rlModelMetrics} scores={modelScores} />
      )}
    </>
  );
}
