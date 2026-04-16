import { useState } from 'react';

const V_COLORS = ["var(--brand-cyan)","var(--brand-coral)","var(--brand-lime)","var(--brand-pink)","var(--brand-gold)","var(--brand-purple)"];

export function RouteTable({ routes }) {
  const [selectedVehicle, setSelectedVehicle] = useState(null);

  if (!routes?.length) {
    return (
      <div style={{ padding: "32px", textAlign: "center", color: "var(--text-muted)" }}>
        Run optimization to view routes.
      </div>
    );
  }

  const visible = selectedVehicle !== null ? routes.filter((_,i)=>i===selectedVehicle) : routes.filter(v=>v.stops?.length);

  return (
    <div style={{ padding: "0 32px 32px", display: "flex", flexDirection: "column", gap: "12px" }}>
      {/* Legend */}
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "8px" }}>
        <button 
          className="btn-primary" 
          onClick={() => setSelectedVehicle(null)} 
          style={{ background: selectedVehicle === null ? "rgba(0,229,255,0.15)" : "transparent" }}
        >
          ALL VEHICLES
        </button>
        {(routes||[]).filter(v=>v.stops?.length).map((v,vi)=>(
          <button 
            key={vi} 
            onClick={() => setSelectedVehicle(selectedVehicle === vi ? null : vi)} 
            className="btn-primary"
            style={{ 
              borderColor: selectedVehicle === vi ? V_COLORS[vi%V_COLORS.length] : "var(--border-light)",
              color: selectedVehicle === vi ? V_COLORS[vi%V_COLORS.length] : "var(--text-muted)",
              background: selectedVehicle === vi ? "rgba(255,255,255,0.05)" : "transparent"
            }}
          >
            V{vi + 1}
          </button>
        ))}
      </div>

      {visible.map((v, idx) => {
        const vi  = selectedVehicle !== null ? selectedVehicle : idx;
        const col = V_COLORS[vi%V_COLORS.length];
        const dist = (v.stops||[]).reduce((s,o)=>s+(o.distance_km||0),0);
        const time = (v.stops||[]).reduce((s,o)=>s+(o.eta_min||0),0);
        const fuel = (v.stops||[]).reduce((s,o)=>s+(o.fuel_L||0),0);

        return (
          <div key={vi} className="glass-panel" style={{ overflow: "hidden" }}>
            <div 
              style={{ padding: "16px 24px", background: "rgba(0,0,0,0.3)", display: "flex", alignItems: "center", gap: "16px", cursor: "pointer" }}
              onClick={() => setSelectedVehicle(selectedVehicle === vi ? null : vi)}
            >
              <div style={{ width: "12px", height: "12px", borderRadius: "2px", background: col, boxShadow: `0 0 10px ${col}` }} />
              <span style={{ fontWeight: 600, fontSize: "14px", color: "var(--text-active)" }}>Vehicle {vi+1}</span>
              <span className="mono-text" style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                {v.stops.length} stops • {dist.toFixed(2)} km • {time.toFixed(0)} min
              </span>
            </div>
            
            {selectedVehicle === vi && (
              <div style={{ padding: "16px", background: "rgba(0,0,0,0.1)", borderTop: "1px solid var(--border-light)" }}>
                <table>
                  <thead>
                    <tr>
                      {["#", "ORDER", "DISTANCE", "ETA", "TRAFFIC", "WEATHER", "C02 EMIT"].map(h => <th key={h}>{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {v.stops.map((s, si) => (
                      <tr key={si} className="mono-text animate-fade-in" style={{ animationDelay: `${si*0.05}s` }}>
                        <td style={{ color: "var(--text-muted)" }}>{si+1}</td>
                        <td style={{ color: col, fontWeight: 600 }}>#{s.order_id || "..." }</td>
                        <td style={{ color: "var(--brand-cyan)" }}>{s.distance_km?.toFixed(2)} km</td>
                        <td style={{ color: "var(--brand-gold)" }}>{s.eta_min?.toFixed(1)} min</td>
                        <td style={{ color: s.traffic > 1 ? "var(--brand-coral)" : "var(--brand-lime)" }}>{s.traffic||"—"}</td>
                        <td style={{ color: "var(--brand-purple)" }}>{s.weather||"—"}</td>
                        <td style={{ color: "var(--text-muted)" }}>{s.co2_kg?.toFixed(2)} kg</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
