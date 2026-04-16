import { useEffect, useRef } from 'react';
import { Play } from 'lucide-react';

const BBOX = { minLat: 20.15, maxLat: 20.40, minLon: 85.70, maxLon: 85.95 };
const denorm = (nLat, nLon) => [
  BBOX.minLat + nLat * (BBOX.maxLat - BBOX.minLat),
  BBOX.minLon + nLon * (BBOX.maxLon - BBOX.minLon),
];

const V_COLORS = ["#00E5FF","#FF6B35","#7CFC00","#FF3CAC","#FFD700","#B39DDB","#80CBC4"];

export function MapTab({ routes, orders, depot, selectedVehicle, onEvent, disabled, log, setLeafletOk }) {
  const containerRef = useRef(null);
  const mapRef       = useRef(null);
  const layersRef    = useRef({ orders:null, routes:[], depot:null });

  useEffect(() => {
    const chk = () => {
      if(window.L) {
        setLeafletOk(true);
        if(!mapRef.current && containerRef.current) {
          const center = denorm(0.7769, 0.8984); // Bhubaneswar roughly
          const map = window.L.map(containerRef.current, { center, zoom:12, zoomControl:true });
          
          // Use a dark modern map tile
          window.L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: '&copy; CartoDB',
            maxZoom: 19,
          }).addTo(map);
          mapRef.current = map;
        }
        return;
      }
      setTimeout(chk, 200);
    }; 
    chk();
    
    return () => {
      if(mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [setLeafletOk]);

  // Handle Orders Drawing
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !window.L) return;
    if (layersRef.current.orders) layersRef.current.orders.clearLayers();
    else layersRef.current.orders = window.L.layerGroup().addTo(map);

    (orders||[]).forEach(o => {
      const p = denorm(o.pickup_lat, o.pickup_lon);
      window.L.circleMarker(p, { radius:3, color:"#00E5FF", fillColor:"#00E5FF", fillOpacity:0.5, weight:1, opacity:0.45 })
        .addTo(layersRef.current.orders);
    });
  }, [orders]);

  // Handle Depot Drawing
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !window.L || !depot) return;
    if (layersRef.current.depot) layersRef.current.depot.remove();
    const pos = denorm(depot.lat, depot.lon);
    const icon = window.L.divIcon({
      className:"",
      html:`<div style="width:18px;height:18px;border-radius:50%;background:#FFD700;border:3px solid #000;box-shadow:0 0 14px #FFD700bb;"></div>`,
      iconAnchor:[9,9],
    });
    layersRef.current.depot = window.L.marker(pos, { icon }).addTo(map);
  }, [depot]);

  // Handle Routes Drawing
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !window.L) return;
    layersRef.current.routes.forEach(l => l.remove());
    layersRef.current.routes = [];

    (routes||[]).forEach((vehicle, vi) => {
      const stops = vehicle.stops || [];
      if (!stops.length) return;
      const col = V_COLORS[vi % V_COLORS.length];
      const dim = selectedVehicle !== null && selectedVehicle !== vi;
      const op  = dim ? 0.12 : 0.88;
      const grp = window.L.layerGroup().addTo(map);

      // 1. Draw the markers immediately (Pickups and Drops)
      stops.forEach((s, si) => {
        const pPos = denorm(s.pickup_lat, s.pickup_lon);
        const dPos = denorm(s.drop_lat,   s.drop_lon);
        
        const pIcon = window.L.divIcon({
          className:"",
          html:`<div style="width:12px;height:12px;border-radius:50%;background:${col};border:2px solid #000;box-shadow:0 0 10px ${col};opacity:${op}"></div>`,
          iconAnchor:[6,6],
        });
        window.L.marker(pPos, { icon:pIcon }).addTo(grp);
      });

      // 2. Async fetch for actual road tracing using OSRM
      async function drawPath() {
        const waypoints = [];
        if (depot) {
            const dp = denorm(depot.lat, depot.lon);
            waypoints.push([dp[1], dp[0]]); // OSRM takes lon, lat
        }
        stops.forEach(s => {
           const pPos = denorm(s.pickup_lat, s.pickup_lon);
           const dPos = denorm(s.drop_lat,   s.drop_lon);
           waypoints.push([pPos[1], pPos[0]]);
           waypoints.push([dPos[1], dPos[0]]);
        });
        if (depot && stops.length) {
            const dp = denorm(depot.lat, depot.lon);
            waypoints.push([dp[1], dp[0]]);
        }

        let isOsrmSuccess = false;
        
        // OSRM public server allows max ~100 coords. We try fetching.
        if (waypoints.length <= 100 && waypoints.length > 1) {
            try {
               const coordString = waypoints.map(w => `${w[0].toFixed(5)},${w[1].toFixed(5)}`).join(';');
               const url = `https://router.project-osrm.org/route/v1/driving/${coordString}?overview=full&geometries=geojson`;
               const res = await fetch(url);
               if (res.ok) {
                  const data = await res.json();
                  if (data.routes && data.routes.length > 0) {
                     const coords = data.routes[0].geometry.coordinates.map(c => [c[1], c[0]]); // GeoJSON is lon,lat -> leaflet is lat,lon
                     window.L.polyline(coords, { color:col, weight:3, opacity:op, className: "animate-fade-in" }).addTo(grp);
                     isOsrmSuccess = true;
                  }
               }
            } catch (e) {
               console.warn("OSRM routing failed, falling back to straight lines.", e);
            }
        }

        // Fallback: draw straight lines if OSRM failed or rate limited limit exceeded
        if (!isOsrmSuccess) {
            if (depot) {
                const dp = denorm(depot.lat, depot.lon);
                const fp = denorm(stops[0].pickup_lat, stops[0].pickup_lon);
                window.L.polyline([dp, fp], { color:col, weight:2, opacity:op*0.4, dashArray:"5 4" }).addTo(grp);
            }
            stops.forEach((s, si) => {
                const pPos = denorm(s.pickup_lat, s.pickup_lon);
                const dPos = denorm(s.drop_lat,   s.drop_lon);
                window.L.polyline([pPos, dPos], { color:col, weight:3, opacity:op }).addTo(grp);
                if (si < stops.length-1) {
                    const nxP = denorm(stops[si+1].pickup_lat, stops[si+1].pickup_lon);
                    window.L.polyline([dPos, nxP], { color:col, weight:2, opacity:op*0.55, dashArray:"4 3" }).addTo(grp);
                }
            });
            if (depot && stops.length) {
                const ld = denorm(stops[stops.length-1].drop_lat, stops[stops.length-1].drop_lon);
                const dp = denorm(depot.lat, depot.lon);
                window.L.polyline([ld, dp], { color:col, weight:2, opacity:op*0.35, dashArray:"3 5" }).addTo(grp);
            }
        }
      }
      
      drawPath();
      layersRef.current.routes.push(grp);
    });

    if (routes?.some(v=>v.stops?.length)) {
      const pts = (routes||[]).flatMap(v=>(v.stops||[]).flatMap(s=>[
        denorm(s.pickup_lat,s.pickup_lon), denorm(s.drop_lat,s.drop_lon)
      ]));
      if (pts.length) map.fitBounds(pts, { padding:[30,30] });
    }
  }, [routes, selectedVehicle, depot]);

  return (
    <div style={{ padding: "0 32px 32px", display: "grid", gridTemplateColumns: "1fr 300px", gap: "24px" }}>
      <div className="glass-panel" style={{ height: "60vh", padding: "8px", position: "relative" }}>
        <div ref={containerRef} style={{ width: "100%", height: "100%", borderRadius: "12px", background: "var(--bg-deep)" }} />
      </div>
      
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div className="glass-panel" style={{ padding: "20px" }}>
          <h3 className="mono-text" style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "16px" }}>EVENT SYSTEM</h3>
          <button className="btn-primary" onClick={() => onEvent("TRAFFIC_UPDATE")} disabled={disabled} style={{ width: "100%", justifyContent: "center", marginBottom: "8px" }}>
            <Play size={14} /> TRIGGER TRAFFIC BURST
          </button>
          <button className="btn-primary" onClick={() => onEvent("NEW_ORDER")} disabled={disabled} style={{ width: "100%", justifyContent: "center" }}>
            <Play size={14} /> INJECT URGENT ORDER
          </button>
        </div>
        
        <div className="glass-panel" style={{ padding: "20px", flex: 1, display: "flex", flexDirection: "column" }}>
          <h3 className="mono-text" style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "16px" }}>SYSTEM LOG</h3>
          <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "6px" }}>
            {log.map((entry, idx) => (
              <div key={idx} className="mono-text animate-fade-in" style={{ fontSize: "10px", color: "rgba(255,255,255,0.7)", background: "rgba(0,0,0,0.3)", padding: "4px 8px", borderRadius: "4px" }}>
                {entry}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
