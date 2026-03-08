import L from "leaflet";
import "leaflet/dist/leaflet.css";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({ iconRetinaUrl: markerIcon2x, iconUrl: markerIcon, shadowUrl: markerShadow });
import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell,
  PieChart, Pie
} from "recharts";

const API = "http://localhost:8000";

// Map normalised [0,1] coords onto Bhubaneswar, Odisha bounding box
const BBOX = { minLat: 20.15, maxLat: 20.40, minLon: 85.70, maxLon: 85.95 };
const denorm = (nLat, nLon) => [
  BBOX.minLat + nLat * (BBOX.maxLat - BBOX.minLat),
  BBOX.minLon + nLon * (BBOX.maxLon - BBOX.minLon),
];

const V_COLORS = ["#00E5FF","#FF6B35","#7CFC00","#FF3CAC","#FFD700","#B39DDB","#80CBC4"];
const TC = { Low:"#7CFC00", Medium:"#FFD700", High:"#FF6B35", Very_High:"#FF3CAC" };
const WC = ["#B0BEC5","#90A4AE","#64B5F6","#1E88E5","#78909C","#FF3CAC","#E0F7FA"];

const apiFetch = async (path, opts={}) => {
  const r = await fetch(API+path, { headers:{"Content-Type":"application/json"}, ...opts });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};
const fmt = (v, d=2) => (typeof v==="number" ? v.toFixed(d) : "—");

// ─────────────────────────────────────────────────────────────────────────────
// REAL MAP using Leaflet + OpenStreetMap (FREE, no API key)
// ─────────────────────────────────────────────────────────────────────────────
function LeafletMap({ routes, orders, depot, selectedVehicle }) {
  const containerRef = useRef(null);
  const mapRef       = useRef(null);
  const layersRef    = useRef({ orders:null, routes:[], depot:null });

  useEffect(() => {
    if (mapRef.current || !L || !containerRef.current) return;
    const center = denorm(0.7769, 0.8984);
    const map = L.map(containerRef.current, { center, zoom:12, zoomControl:true });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 19,
    }).addTo(map);
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  // Draw all order pickup/drop dots
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !L) return;
    if (layersRef.current.orders) layersRef.current.orders.clearLayers();
    else layersRef.current.orders = L.layerGroup().addTo(map);

    (orders||[]).forEach(o => {
      const p = denorm(o.pickup_lat, o.pickup_lon);
      const d = denorm(o.drop_lat,   o.drop_lon);
      L.circleMarker(p, { radius:3, color:"#00E5FF", fillColor:"#00E5FF", fillOpacity:0.5, weight:1, opacity:0.45 })
        .bindPopup(`<b style="font-family:monospace">Order #${o.order_id??""}</b><br>📦 Pickup<br>🚦 ${o.traffic_label??o.Road_traffic_density}<br>🌦 ${o.weather_label??o.Weather_conditions}`)
        .addTo(layersRef.current.orders);
      L.circleMarker(d, { radius:3, color:"#fff", fillColor:"#fff", fillOpacity:0.3, weight:1, opacity:0.3 })
        .bindPopup(`<b style="font-family:monospace">Order #${o.order_id??""}</b><br>🏠 Drop<br>📍 ${fmt(o.distance_km)} km`)
        .addTo(layersRef.current.orders);
    });
  }, [orders]);

  // Draw depot star
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !L || !depot) return;
    if (layersRef.current.depot) layersRef.current.depot.remove();
    const pos = denorm(depot.lat, depot.lon);
    const icon = L.divIcon({
      className:"",
      html:`<div style="width:18px;height:18px;border-radius:50%;background:#FFD700;border:3px solid #fff;box-shadow:0 0 14px #FFD700bb;"></div>`,
      iconAnchor:[9,9],
    });
    layersRef.current.depot = L.marker(pos, { icon })
      .bindPopup("<b style='font-family:monospace'>🏭 Virtual Depot</b><br>Centroid of all pickups")
      .addTo(map);
  }, [depot]);

  // Draw vehicle routes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !L) return;
    layersRef.current.routes.forEach(l => l.remove());
    layersRef.current.routes = [];

    (routes||[]).forEach((vehicle, vi) => {
      const stops = vehicle.stops || [];
      if (!stops.length) return;
      const col = V_COLORS[vi % V_COLORS.length];
      const dim = selectedVehicle !== null && selectedVehicle !== vi;
      const op  = dim ? 0.12 : 0.88;
      const grp = L.layerGroup().addTo(map);

      if (depot) {
        const dp = denorm(depot.lat, depot.lon);
        const fp = denorm(stops[0].pickup_lat, stops[0].pickup_lon);
        L.polyline([dp, fp], { color:col, weight:1.5, opacity:op*0.4, dashArray:"5 4" }).addTo(grp);
      }

      stops.forEach((s, si) => {
        const pPos = denorm(s.pickup_lat, s.pickup_lon);
        const dPos = denorm(s.drop_lat,   s.drop_lon);
        L.polyline([pPos, dPos], { color:col, weight:2.5, opacity:op }).addTo(grp);
        if (si < stops.length-1) {
          const nxP = denorm(stops[si+1].pickup_lat, stops[si+1].pickup_lon);
          L.polyline([dPos, nxP], { color:col, weight:1.5, opacity:op*0.55, dashArray:"4 3" }).addTo(grp);
        }
        const pIcon = L.divIcon({
          className:"",
          html:`<div style="width:11px;height:11px;border-radius:50%;background:${col};border:2px solid #fff;box-shadow:0 0 7px ${col}99;opacity:${op}"></div>`,
          iconAnchor:[5.5,5.5],
        });
        L.marker(pPos, { icon:pIcon })
          .bindPopup(`<div style="font-family:monospace;font-size:12px;min-width:180px"><b style="color:${col}">V${vi+1} Stop ${si+1}</b><br>Order #${s.order_id??""}<br>📍 ${fmt(s.distance_km)} km &nbsp;⏱ ${fmt(s.eta_min,0)} min<br>🚦 ${s.traffic||"—"} &nbsp;🌦 ${s.weather||"—"}</div>`)
          .addTo(grp);
        const dIcon = L.divIcon({
          className:"",
          html:`<div style="width:7px;height:7px;border-radius:50%;background:#fff;border:2px solid ${col};opacity:${op*0.75}"></div>`,
          iconAnchor:[3.5,3.5],
        });
        L.marker(dPos, { icon:dIcon }).addTo(grp);
      });

      if (depot && stops.length) {
        const ld = denorm(stops[stops.length-1].drop_lat, stops[stops.length-1].drop_lon);
        const dp = denorm(depot.lat, depot.lon);
        L.polyline([ld, dp], { color:col, weight:1.2, opacity:op*0.35, dashArray:"3 5" }).addTo(grp);
      }
      layersRef.current.routes.push(grp);
    });

    if (routes?.some(v=>v.stops?.length)) {
      const pts = (routes||[]).flatMap(v=>(v.stops||[]).flatMap(s=>[
        denorm(s.pickup_lat,s.pickup_lon), denorm(s.drop_lat,s.drop_lon)
      ]));
      if (pts.length) map.fitBounds(pts, { padding:[30,30] });
    }
  }, [routes, selectedVehicle, depot]);

  return <div ref={containerRef} style={{ width:"100%", height:"100%", borderRadius:12, overflow:"hidden" }} />;
}

// ─────────────────────────────────────────────────────────────────────────────
// METRIC CARD
// ─────────────────────────────────────────────────────────────────────────────
function Card({ label, value, unit, sub, color="#00E5FF", icon, trend }) {
  return (
    <div style={{ background:"linear-gradient(135deg,#06101F,#08152A)", border:`1px solid ${color}28`, borderRadius:10, padding:"13px 16px", flex:1, minWidth:130, boxShadow:`0 4px 20px ${color}06` }}>
      <div style={{ color:"#4A6077", fontSize:10, letterSpacing:1.5, marginBottom:4, fontFamily:"'Courier New',monospace" }}>{icon} {label}</div>
      <div style={{ color, fontSize:24, fontWeight:700, fontFamily:"'Courier New',monospace", lineHeight:1 }}>
        {value}<span style={{ fontSize:11, color:"#4A6077", marginLeft:4 }}>{unit}</span>
      </div>
      {sub   && <div style={{ color:"#2E4057", fontSize:10, marginTop:3, fontFamily:"monospace" }}>{sub}</div>}
      {trend && <div style={{ color:"#7CFC00", fontSize:10, marginTop:3, fontFamily:"monospace" }}>{trend}</div>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// VEHICLE LEGEND
// ─────────────────────────────────────────────────────────────────────────────
function VehicleLegend({ routes, selected, onSelect }) {
  return (
    <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
      <button onClick={()=>onSelect(null)} style={{ padding:"4px 12px", borderRadius:20, fontSize:11, cursor:"pointer", fontFamily:"monospace", border:`1px solid ${selected===null?"#00E5FF":"#1A2744"}`, background:selected===null?"#00E5FF22":"transparent", color:selected===null?"#00E5FF":"#546E7A" }}>ALL</button>
      {(routes||[]).filter(v=>v.stops?.length).map((v,vi)=>(
        <button key={vi} onClick={()=>onSelect(selected===vi?null:vi)} style={{ padding:"4px 12px", borderRadius:20, fontSize:11, cursor:"pointer", fontFamily:"monospace", border:`1px solid ${selected===vi?V_COLORS[vi%V_COLORS.length]:"#1A2744"}`, background:selected===vi?`${V_COLORS[vi%V_COLORS.length]}22`:"transparent", color:selected===vi?V_COLORS[vi%V_COLORS.length]:"#546E7A", display:"flex", alignItems:"center", gap:5 }}>
          <span style={{ width:8, height:8, borderRadius:2, background:V_COLORS[vi%V_COLORS.length], display:"inline-block" }} />
          V{vi+1} · {v.stops.length} stops
        </button>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ROUTE TABLE
// ─────────────────────────────────────────────────────────────────────────────
function RouteTable({ routes, selectedVehicle, onSelect }) {
  if (!routes?.length) return <div style={{ color:"#2E4057", fontSize:13, padding:24, textAlign:"center" }}>Run optimisation to see routes.</div>;
  const visible = selectedVehicle!==null ? routes.filter((_,i)=>i===selectedVehicle) : routes.filter(v=>v.stops?.length);
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
      {visible.map((v,idx)=>{
        const vi  = selectedVehicle!==null ? selectedVehicle : idx;
        const col = V_COLORS[vi%V_COLORS.length];
        const td  = (v.stops||[]).reduce((s,o)=>s+(o.distance_km||0),0);
        const te  = (v.stops||[]).reduce((s,o)=>s+(o.eta_min||0),0);
        return (
          <div key={vi}>
            <div onClick={()=>onSelect(selectedVehicle===vi?null:vi)} style={{ display:"flex", alignItems:"center", gap:10, padding:"9px 14px", background:"#06101F", borderRadius:8, cursor:"pointer", border:`1px solid ${col}44`, userSelect:"none" }}>
              <div style={{ width:10,height:10,borderRadius:2,background:col,flexShrink:0 }} />
              <span style={{ color:"#B0BEC5",fontFamily:"monospace",fontSize:13,fontWeight:600 }}>Vehicle {vi+1}</span>
              <span style={{ color:"#4A6077",fontSize:11,fontFamily:"monospace" }}>{v.stops.length} stops · {fmt(td)} km · {fmt(te,0)} min · {fmt(td*0.12,2)} L fuel</span>
              <span style={{ marginLeft:"auto",color:"#4A6077",fontSize:11 }}>{selectedVehicle===vi?"▲":"▼"}</span>
            </div>
            {selectedVehicle===vi && (
              <div style={{ background:"#050D1C",border:"1px solid #0D2137",borderTop:"none",borderRadius:"0 0 8px 8px",overflowX:"auto" }}>
                <table style={{ width:"100%",borderCollapse:"collapse",fontSize:11,fontFamily:"monospace" }}>
                  <thead>
                    <tr style={{ background:"#08152A" }}>
                      {["#","ORDER","DIST km","ETA min","TRAFFIC","WEATHER","FUEL L"].map(h=>(
                        <th key={h} style={{ padding:"7px 10px",color:"#4A6077",textAlign:"left",fontWeight:400,whiteSpace:"nowrap" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {v.stops.map((s,si)=>(
                      <tr key={si} style={{ borderTop:"1px solid #0D2137" }} onMouseEnter={e=>e.currentTarget.style.background="#08152A"} onMouseLeave={e=>e.currentTarget.style.background="transparent"}>
                        <td style={{ padding:"6px 10px",color:"#2E4057" }}>{si+1}</td>
                        <td style={{ color:col,fontWeight:600 }}>#{s.order_id??"—"}</td>
                        <td style={{ color:"#00E5FF" }}>{fmt(s.distance_km)}</td>
                        <td style={{ color:"#FFD700" }}>{fmt(s.eta_min,0)}</td>
                        <td style={{ color:TC[s.traffic]||"#B0BEC5" }}>{s.traffic||"—"}</td>
                        <td style={{ color:"#78909C" }}>{s.weather||"—"}</td>
                        <td style={{ color:"#546E7A" }}>{fmt((s.distance_km||0)*0.12,3)}</td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr style={{ borderTop:"2px solid #0D2137",background:"#06101F" }}>
                      <td colSpan={2} style={{ padding:"6px 10px",color:"#4A6077" }}>TOTAL</td>
                      <td style={{ color:"#00E5FF",fontWeight:700 }}>{fmt(td)}</td>
                      <td style={{ color:"#FFD700",fontWeight:700 }}>{fmt(te,0)}</td>
                      <td colSpan={3} style={{ color:"#4A6077",fontSize:10 }}>Fuel: {fmt(td*0.12)} L</td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// EVENT PANEL
// ─────────────────────────────────────────────────────────────────────────────
function EventPanel({ onEvent, disabled, log }) {
  const [type,setType]     = useState("NEW_ORDER");
  const [status,setStatus] = useState(null);
  const [busy,setBusy]     = useState(false);

  const fire = async () => {
    setBusy(true); setStatus(null);
    try {
      let payload = {};
      if (type==="NEW_ORDER") payload={ order:{
        order_id: Math.floor(Math.random()*99999),
        pickup_lat:Math.random()*0.4+0.6, pickup_lon:Math.random()*0.3+0.7,
        drop_lat:Math.random()*0.45+0.5,  drop_lon:Math.random()*0.3+0.7,
        Road_traffic_density:Math.floor(Math.random()*4),
        Weather_conditions:Math.floor(Math.random()*7),
        order_time_min:800+Math.floor(Math.random()*400),
        pickup_time_min:815+Math.floor(Math.random()*400),
      }};
      const res = await onEvent(type, payload);
      setStatus({ ok:true, msg:`${res.strategy} · ${res.solve_time_s}s · ${fmt(res.total_dist_km)} km` });
    } catch(e) { setStatus({ ok:false, msg:e.message }); }
    finally { setBusy(false); }
  };

  return (
    <div style={{ background:"#06101F",border:"1px solid #0D2137",borderRadius:10,padding:16 }}>
      <div style={{ color:"#B0BEC5",fontSize:11,letterSpacing:2,marginBottom:10,fontFamily:"monospace" }}>⚡ REAL-TIME EVENTS</div>
      <div style={{ display:"flex",gap:5,marginBottom:10 }}>
        {[["NEW_ORDER","📦"],["TRAFFIC_UPDATE","🚦"],["DELAY","⏱"]].map(([t,ic])=>(
          <button key={t} onClick={()=>setType(t)} style={{ flex:1,padding:"7px 0",borderRadius:6,cursor:"pointer",fontSize:10,fontFamily:"monospace",letterSpacing:.5,border:`1px solid ${type===t?"#00E5FF":"#1A2744"}`,background:type===t?"#00E5FF18":"transparent",color:type===t?"#00E5FF":"#4A6077" }}>
            {ic} {t.replace("_"," ")}
          </button>
        ))}
      </div>
      <button onClick={fire} disabled={disabled||busy} style={{ width:"100%",padding:"9px 0",borderRadius:8,cursor:disabled||busy?"not-allowed":"pointer",background:disabled||busy?"#0D2137":"linear-gradient(90deg,#00E5FF22,#00E5FF33)",border:"1px solid #00E5FF44",color:disabled||busy?"#2E4057":"#00E5FF",fontFamily:"monospace",fontSize:13,letterSpacing:1 }}>
        {busy?"⏳ PROCESSING…":"▶ FIRE EVENT"}
      </button>
      {status && (
        <div style={{ marginTop:8,padding:"7px 10px",borderRadius:6,fontSize:11,fontFamily:"monospace",background:status.ok?"#7CFC0012":"#FF3CAC12",border:`1px solid ${status.ok?"#7CFC0033":"#FF3CAC33"}`,color:status.ok?"#7CFC00":"#FF3CAC" }}>
          {status.ok?"✅":"❌"} {status.msg}
        </div>
      )}
      {log?.length>0 && (
        <div style={{ marginTop:12 }}>
          <div style={{ color:"#4A6077",fontSize:10,letterSpacing:1.5,marginBottom:6 }}>ACTIVITY LOG</div>
          <div style={{ maxHeight:120,overflowY:"auto",display:"flex",flexDirection:"column",gap:3 }}>
            {log.map((l,i)=>(
              <div key={i} style={{ fontSize:10,fontFamily:"monospace",padding:"3px 6px",borderRadius:4,color:l.includes("❌")?"#FF3CAC":l.includes("✅")?"#7CFC00":"#4A6077",background:"#050D1C" }}>{l}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [orders,setOrders]         = useState([]);
  const [routes,setRoutes]         = useState([]);
  const [metrics,setMetrics]       = useState(null);
  const [stats,setStats]           = useState(null);
  const [rewardData,setRewardData] = useState([]);
  const [trafficData,setTrafficData]=useState([]);
  const [weatherData,setWeatherData]=useState([]);
  const [depot,setDepot]           = useState(null);
  const [loading,setLoading]       = useState(false);
  const [tab,setTab]               = useState("map");
  const [nOrders,setNOrders]       = useState(25);
  const [nVehicles,setNVehicles]   = useState(3);
  const [capacity,setCapacity]     = useState(10);
  const [solveInfo,setSolveInfo]   = useState(null);
  const [log,setLog]               = useState([]);
  const [apiOk,setApiOk]           = useState(null);
  const [leafletOk,setLeafletOk]   = useState(false);
  const [selVehicle,setSelVehicle] = useState(null);

  const addLog = useCallback(msg=>setLog(l=>[`[${new Date().toLocaleTimeString()}] ${msg}`,...l.slice(0,29)]),[]);

  useEffect(()=>{
    const chk=()=>{ if(L){setLeafletOk(true);return;} setTimeout(chk,200); }; chk();
  },[]);

  useEffect(()=>{
    apiFetch("/health").then(()=>{setApiOk(true);addLog("✅ Backend connected");}).catch(()=>{setApiOk(false);addLog("❌ Backend offline");});
    apiFetch("/dataset-stats").then(s=>{
      setStats(s);
      setTrafficData(Object.entries(s.traffic_dist).map(([k,v])=>({name:k,orders:v})));
      setWeatherData(Object.entries(s.weather_dist).map(([k,v])=>({name:k,value:v})));
    }).catch(()=>{});
    apiFetch("/reward-curve").then(d=>{
      setRewardData(d.steps.map((s,i)=>({step:s/1000,reward:d.rewards[i]})));
    }).catch(()=>{});
  },[]);

  const loadOrders = useCallback(async()=>{
    setLoading(true);
    try {
      const res=await apiFetch(`/orders?limit=${nOrders}`);
      setOrders(res.orders);
      addLog(`✅ Loaded ${res.orders.length} orders`);
    } catch(e){addLog(`❌ ${e.message}`);}
    finally{setLoading(false);}
  },[nOrders,addLog]);

  const runOptimize = async()=>{
    if(!orders.length){addLog("Load orders first");return;}
    setLoading(true); setSolveInfo(null);
    try {
      const res=await apiFetch("/optimize",{method:"POST",body:JSON.stringify({orders,n_vehicles:nVehicles,capacity,time_limit:10})});
      setRoutes(res.routes_detail||[]);
      setMetrics({...res.metrics,solve_time:res.solve_time_s});
      setSolveInfo({status:res.status,time:res.solve_time_s,dist:res.total_dist_km});
      setSelVehicle(null);
      const lat=orders.reduce((s,o)=>s+o.pickup_lat,0)/orders.length;
      const lon=orders.reduce((s,o)=>s+o.pickup_lon,0)/orders.length;
      setDepot({lat,lon});
      addLog(`✅ Optimised — ${res.status} · ${res.total_dist_km} km · ${res.solve_time_s}s`);
    } catch(e){addLog(`❌ ${e.message}`);}
    finally{setLoading(false);}
  };

  const handleEvent=async(type,payload)=>{
    const res=await apiFetch("/event",{method:"POST",body:JSON.stringify({event_type:type,payload})});
    setRoutes(res.routes_detail||[]);
    setMetrics(m=>({...m,...res.metrics}));
    addLog(`⚡ ${type} → ${res.strategy} · ${res.solve_time_s}s`);
    return res;
  };

  const S = {
    app:{ minHeight:"100vh",background:"#030912",color:"#B0BEC5",fontFamily:"'Courier New',monospace",display:"flex",flexDirection:"column" },
    hdr:{ background:"linear-gradient(180deg,#050D1A,#030912)",borderBottom:"1px solid #0D2137",padding:"14px 28px",display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0 },
    ctrl:{ padding:"12px 28px",display:"flex",gap:12,alignItems:"center",flexWrap:"wrap",borderBottom:"1px solid #0D2137",flexShrink:0 },
    tabs:{ padding:"0 28px",borderBottom:"1px solid #0D2137",display:"flex",gap:0,flexShrink:0 },
    tb:(t)=>({ padding:"8px 18px",cursor:"pointer",fontSize:11,letterSpacing:1.5,border:"none",borderBottom:tab===t?"2px solid #00E5FF":"2px solid transparent",background:"transparent",color:tab===t?"#00E5FF":"#4A6077",fontFamily:"'Courier New',monospace" }),
    inp:{ background:"#06101F",border:"1px solid #0D2137",borderRadius:6,color:"#00E5FF",fontFamily:"monospace",fontSize:13,padding:"6px 10px",width:62,textAlign:"center" },
    btn:(c="#00E5FF",d=false)=>({ padding:"8px 18px",borderRadius:8,border:`1px solid ${d?"#1A2744":c+"55"}`,background:d?"#06101F":`${c}18`,color:d?"#2E4057":c,cursor:d?"not-allowed":"pointer",fontFamily:"monospace",fontSize:12,letterSpacing:1 }),
  };

  return (
    <div style={S.app}>
      {/* HEADER */}
      <div style={S.hdr}>
        <div>
          <div style={{ color:"#00E5FF",fontSize:17,fontWeight:700,letterSpacing:3 }}>◈ DELIVERY ROUTE OPTIMISER</div>
          <div style={{ color:"#2E4057",fontSize:9,letterSpacing:4,marginTop:2 }}>HYBRID DRL + OR-TOOLS · REAL-TIME · BHUBANESWAR, ODISHA</div>
        </div>
        <div style={{ display:"flex",alignItems:"center",gap:16 }}>
          {[["MAP",leafletOk],["API",apiOk]].map(([lbl,ok])=>(
            <div key={lbl} style={{ display:"flex",alignItems:"center",gap:5 }}>
              <div style={{ width:7,height:7,borderRadius:"50%",background:ok===null?"#FFD700":ok?"#7CFC00":"#FF3CAC",boxShadow:`0 0 6px ${ok?"#7CFC00":ok===null?"#FFD700":"#FF3CAC"}` }} />
              <span style={{ color:"#4A6077",fontSize:10 }}>{lbl} {ok===null?"…":ok?"READY":"OFFLINE"}</span>
            </div>
          ))}
          {solveInfo && <div style={{ color:"#4A6077",fontSize:10 }}>{solveInfo.status} · {solveInfo.time}s</div>}
        </div>
      </div>

      {apiOk===false && (
        <div style={{ margin:"12px 28px 0",padding:"10px 16px",borderRadius:8,background:"#FF3CAC0A",border:"1px solid #FF3CAC28",color:"#FF3CAC",fontSize:11 }}>
          ⚠ Backend offline — run: <code style={{ background:"#0D0D0D",padding:"1px 6px",borderRadius:4 }}>cd backend && uvicorn main:app --reload</code>
        </div>
      )}

      {/* CONTROLS */}
      <div style={S.ctrl}>
        {[["ORDERS",nOrders,setNOrders,5,200],["VEHICLES",nVehicles,setNVehicles,1,7],["CAPACITY",capacity,setCapacity,1,50]].map(([lbl,val,set,mn,mx])=>(
          <div key={lbl} style={{ display:"flex",alignItems:"center",gap:6 }}>
            <span style={{ color:"#4A6077",fontSize:10,letterSpacing:1 }}>{lbl}</span>
            <input type="number" value={val} min={mn} max={mx} onChange={e=>set(+e.target.value)} style={S.inp} />
          </div>
        ))}
        <div style={{ width:1,height:26,background:"#0D2137" }} />
        <button onClick={loadOrders} disabled={loading} style={S.btn("#00E5FF",loading)}>{loading?"⏳":"📂"} LOAD ORDERS</button>
        <button onClick={runOptimize} disabled={loading||!orders.length} style={S.btn("#7CFC00",loading||!orders.length)}>{loading?"⏳":"▶"} OPTIMISE</button>
        {orders.length>0 && <span style={{ color:"#2E4057",fontSize:10 }}>{orders.length} orders ready</span>}
      </div>

      {/* METRICS STRIP */}
      {metrics && (
        <div style={{ padding:"12px 28px",display:"flex",gap:10,flexWrap:"wrap",borderBottom:"1px solid #0D2137" }}>
          <Card label="DISTANCE"     value={fmt(metrics.total_dist_km)}    unit="km"  color="#00E5FF" icon="📍" trend={`↓ ${(((43.3-metrics.total_dist_km)/43.3)*100).toFixed(1)}% vs baseline`} />
          <Card label="TIME"         value={fmt(metrics.total_time_min,0)} unit="min" color="#FFD700" icon="⏱" />
          <Card label="FUEL"         value={fmt(metrics.total_fuel_L)}     unit="L"   color="#FF6B35" icon="⛽" />
          <Card label="ORDERS"       value={metrics.orders_served}         unit=""    color="#7CFC00" icon="📦" />
          <Card label="ON-TIME"      value={metrics.on_time_pct}           unit="%"   color="#FF3CAC" icon="✅" />
          <Card label="VEHICLES"     value={metrics.vehicles_used}         unit=""    color="#B39DDB" icon="🚚" />
          <Card label="SOLVE TIME"   value={fmt(metrics.solve_time,2)}     unit="s"   color="#80CBC4" icon="⚡" sub="RL warm-start + OR-Tools" />
        </div>
      )}

      {/* TABS */}
      <div style={S.tabs}>
        {[["map","🗺 MAP"],["routes","🚚 ROUTES"],["analytics","📊 ANALYTICS"],["dataset","🧹 DATASET"]].map(([t,l])=>(
          <button key={t} onClick={()=>setTab(t)} style={S.tb(t)}>{l}</button>
        ))}
      </div>

      {/* ── MAP TAB ─────────────────────────────────────────────────────────── */}
      {tab==="map" && (
        <div style={{ flex:1,padding:"14px 28px",display:"grid",gridTemplateColumns:"1fr 310px",gap:16,minHeight:0 }}>
          <div style={{ display:"flex",flexDirection:"column",gap:8,minHeight:0 }}>
            <VehicleLegend routes={routes} selected={selVehicle} onSelect={setSelVehicle} />
            <div style={{ flex:1,minHeight:500,borderRadius:12,overflow:"hidden",border:"1px solid #0D2137" }}>
              {leafletOk
                ? <LeafletMap routes={routes} orders={orders} depot={depot} selectedVehicle={selVehicle} />
                : <div style={{ height:"100%",display:"flex",alignItems:"center",justifyContent:"center",color:"#2E4057",fontSize:13 }}>Loading map…</div>}
            </div>
          </div>
          <div style={{ display:"flex",flexDirection:"column",gap:14,overflow:"auto" }}>
            <EventPanel onEvent={handleEvent} disabled={!routes.length} log={log} />
            <div style={{ background:"#06101F",border:"1px solid #0D2137",borderRadius:10,padding:16 }}>
              <div style={{ color:"#B0BEC5",fontSize:11,letterSpacing:2,marginBottom:10 }}>📊 VS BASELINE</div>
              <table style={{ width:"100%",fontSize:11,borderCollapse:"collapse",fontFamily:"monospace" }}>
                <thead><tr>{["Metric","OR-Tools","Hybrid","Δ"].map(h=><th key={h} style={{ padding:"4px 8px",color:"#4A6077",textAlign:h==="Δ"?"right":"left",fontWeight:400 }}>{h}</th>)}</tr></thead>
                <tbody>
                  {[["Distance","43.3 km",metrics?`${fmt(metrics.total_dist_km)} km`:"—",metrics?`${(((43.3-metrics.total_dist_km)/43.3)*100).toFixed(1)}%`:"↓ 5.8%"],["Time","104 min",metrics?`${fmt(metrics.total_time_min,0)} min`:"—","↓ 5.8%"],["Fuel","5.2 L",metrics?`${fmt(metrics.total_fuel_L)} L`:"—","↓ 5.8%"],["Solve","12 s",metrics?`${fmt(metrics.solve_time,1)} s`:"—","↓ 58%"],["On-time","92%",metrics?`${metrics.on_time_pct}%`:"—","↑ 2%"]].map(([m,a,b,d])=>(
                    <tr key={m} style={{ borderTop:"1px solid #0D2137" }}>
                      <td style={{ padding:"5px 8px",color:"#78909C" }}>{m}</td>
                      <td style={{ color:"#4A6077" }}>{a}</td>
                      <td style={{ color:"#00E5FF" }}>{b}</td>
                      <td style={{ textAlign:"right",fontWeight:700,color:"#7CFC00" }}>{d}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── ROUTES TAB ──────────────────────────────────────────────────────── */}
      {tab==="routes" && (
        <div style={{ padding:"14px 28px" }}>
          {routes.length>0 && <div style={{ marginBottom:10 }}><VehicleLegend routes={routes} selected={selVehicle} onSelect={setSelVehicle} /></div>}
          <RouteTable routes={routes} selectedVehicle={selVehicle} onSelect={setSelVehicle} />
        </div>
      )}

      {/* ── ANALYTICS TAB ───────────────────────────────────────────────────── */}
      {tab==="analytics" && (
        <div style={{ padding:"14px 28px",display:"grid",gridTemplateColumns:"1fr 1fr",gap:16 }}>
          <div style={{ background:"#06101F",border:"1px solid #0D2137",borderRadius:10,padding:16 }}>
            <div style={{ color:"#B0BEC5",fontSize:11,letterSpacing:2,marginBottom:12 }}>📈 RL REWARD CURVE (PPO · 50k–200k steps)</div>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={rewardData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#0D2137" />
                <XAxis dataKey="step" stroke="#4A6077" fontSize={10} />
                <YAxis stroke="#4A6077" fontSize={10} />
                <Tooltip contentStyle={{ background:"#050D1C",border:"1px solid #0D2137",color:"#B0BEC5",fontSize:11,fontFamily:"monospace" }} />
                <Line type="monotone" dataKey="reward" stroke="#00E5FF" strokeWidth={2} dot={{ r:3,fill:"#00E5FF",strokeWidth:0 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div style={{ background:"#06101F",border:"1px solid #0D2137",borderRadius:10,padding:16 }}>
            <div style={{ color:"#B0BEC5",fontSize:11,letterSpacing:2,marginBottom:12 }}>🚦 TRAFFIC DISTRIBUTION</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={trafficData} barCategoryGap="35%">
                <CartesianGrid strokeDasharray="3 3" stroke="#0D2137" />
                <XAxis dataKey="name" stroke="#4A6077" fontSize={10} />
                <YAxis stroke="#4A6077" fontSize={10} />
                <Tooltip contentStyle={{ background:"#050D1C",border:"1px solid #0D2137",color:"#B0BEC5",fontSize:11,fontFamily:"monospace" }} />
                <Bar dataKey="orders" radius={[4,4,0,0]}>
                  {trafficData.map((e,i)=><Cell key={i} fill={TC[e.name]||"#546E7A"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ background:"#06101F",border:"1px solid #0D2137",borderRadius:10,padding:16 }}>
            <div style={{ color:"#B0BEC5",fontSize:11,letterSpacing:2,marginBottom:12 }}>🌦 WEATHER DISTRIBUTION</div>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={weatherData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75} label={({name,percent})=>`${name} ${(percent*100).toFixed(0)}%`} labelLine={{ stroke:"#0D2137" }}>
                  {weatherData.map((_,i)=><Cell key={i} fill={WC[i%WC.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background:"#050D1C",border:"1px solid #0D2137",color:"#B0BEC5",fontSize:11,fontFamily:"monospace" }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={{ background:"#06101F",border:"1px solid #0D2137",borderRadius:10,padding:16 }}>
            <div style={{ color:"#B0BEC5",fontSize:11,letterSpacing:2,marginBottom:12 }}>⚡ DYNAMIC RE-OPT (Paper Results)</div>
            <table style={{ width:"100%",fontSize:12,borderCollapse:"collapse",fontFamily:"monospace" }}>
              <thead><tr style={{ color:"#4A6077" }}><th style={{ textAlign:"left",padding:"5px 8px" }}>Scenario</th><th>Dist</th><th>Time</th><th>Fuel</th></tr></thead>
              <tbody>
                {[["OR-Tools baseline","43.3 km","104 min","5.2 L","#4A6077"],["Hybrid (optimal)","40.8 km","98 min","4.9 L","#7CFC00"],["After new order","45.0 km","108 min","5.4 L","#FF6B35"]].map(([s,d,t,f,c])=>(
                  <tr key={s} style={{ borderTop:"1px solid #0D2137" }}>
                    <td style={{ padding:"6px 8px",color:c }}>{s}</td>
                    <td style={{ textAlign:"center",color:"#78909C" }}>{d}</td>
                    <td style={{ textAlign:"center",color:"#78909C" }}>{t}</td>
                    <td style={{ textAlign:"center",color:"#78909C" }}>{f}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DATASET TAB ─────────────────────────────────────────────────────── */}
      {tab==="dataset" && (
        <div style={{ padding:"14px 28px" }}>
          {stats ? (
            <div style={{ display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(155px,1fr))",gap:10,marginBottom:18 }}>
              <Card label="TOTAL ORDERS"   value={stats.total_orders.toLocaleString()} unit=""    color="#00E5FF" icon="📦" />
              <Card label="AVG DISTANCE"   value={fmt(stats.avg_distance_km)}          unit="km"  color="#FFD700" icon="📍" />
              <Card label="AVG TIME TAKEN" value={fmt(stats.avg_time_taken_min,1)}      unit="min" color="#FF6B35" icon="⏱" />
              <Card label="AVG WAIT TIME"  value={fmt(stats.avg_wait_time_min,1)}       unit="min" color="#7CFC00" icon="⌛" />
              <Card label="AVG FUEL"       value={fmt(stats.avg_fuel_L,3)}              unit="L"   color="#B39DDB" icon="⛽" />
              <Card label="DEPOT"          value={`${fmt(stats.depot?.lat,3)}, ${fmt(stats.depot?.lon,3)}`} unit="" color="#FF3CAC" icon="🏭" sub="Bhubaneswar centroid" />
            </div>
          ) : <div style={{ color:"#2E4057",marginBottom:18 }}>Loading…</div>}
          <div style={{ background:"#06101F",border:"1px solid #0D2137",borderRadius:10,padding:18 }}>
            <div style={{ color:"#B0BEC5",fontSize:11,letterSpacing:2,marginBottom:14 }}>🧹 PREPROCESSING PIPELINE</div>
            {[["01","LOAD","45,584 raw rows ingested","#00E5FF"],["02","DEDUP","31 exact duplicate rows dropped","#7CFC00"],["03","TIMING FIX","6,260 rows where pickup_time < order_time — columns swapped","#FFD700"],["04","OUTLIER RM","5,560 extreme wait-time rows (>120 min) removed","#FF6B35"],["05","WINSORISE","431 distance + 270 time_taken outliers capped via IQR","#B39DDB"],["06","ETA","Traffic-adjusted ETA: dist/25kmh × traffic_factor","#00E5FF"],["07","FUEL","Fuel estimate: 0.12 L/km","#7CFC00"],["08","DEPOT","Virtual depot = centroid of all pickup coords","#FFD700"],["09","FEATURES","Normalised time-of-day + depot-to-pickup distance added","#FF3CAC"],["✅","OUTPUT","39,993 rows × 22 columns → orders_clean.csv","#7CFC00"]].map(([n,tag,desc,col])=>(
              <div key={n} style={{ display:"flex",gap:12,padding:"7px 0",borderBottom:"1px solid #0D2137",alignItems:"flex-start" }}>
                <div style={{ color:col,fontFamily:"monospace",fontSize:10,width:22,flexShrink:0,paddingTop:1 }}>{n}</div>
                <div style={{ color:col,fontFamily:"monospace",fontSize:10,width:78,flexShrink:0,paddingTop:1,letterSpacing:1 }}>{tag}</div>
                <div style={{ color:"#78909C",fontSize:12 }}>{desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/*
────────────────────────────────────────────────────────────────────────────────
 index.html SETUP — add inside <head>, BEFORE the React <script> tag:

   <!-- Leaflet CSS (free OpenStreetMap, no API key needed) -->
   <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
   <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
────────────────────────────────────────────────────────────────────────────────
*/
