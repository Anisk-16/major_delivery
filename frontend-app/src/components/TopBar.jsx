import { Activity, Server, Map, Settings } from 'lucide-react';

export function TopBar({ apiOk, leafletOk, wsStatus, schedRunning, schedInterval, solveInfo }) {
  return (
    <div style={{ padding: "16px 32px", borderBottom: "1px solid var(--border-light)", display: "flex", justifyContent: "space-between", alignItems: "center", background: "rgba(0,0,0,0.2)" }}>
      <div>
        <div className="title-gradient" style={{ fontSize: "20px", fontWeight: 800, letterSpacing: "2px" }}>
          ◈ DRO :: HYBRID ENGINE
        </div>
        <div className="mono-text" style={{ color: "var(--text-muted)", fontSize: "10px", marginTop: "4px", letterSpacing: "3px" }}>
          DEEP RL + OR-TOOLS • BHUBANESWAR
        </div>
      </div>

      <div style={{ display: "flex", gap: "24px", alignItems: "center" }}>
        
        {solveInfo && (
          <div className="glass-panel mono-text" style={{ padding: "6px 12px", fontSize: "11px", color: "var(--brand-cyan)", display: "flex", alignItems: "center", gap: "6px", border: "1px solid var(--border-focus)" }}>
            <Activity size={12} />
            {solveInfo.status} • {solveInfo.time}s
          </div>
        )}

        <StatusBadge icon={Server} label="API" status={apiOk} />
        <StatusBadge icon={Map} label="MAP" status={leafletOk} />
        <StatusBadge 
          icon={Activity} 
          label="WS" 
          status={wsStatus === "connected" ? true : wsStatus === "error" ? false : null} 
        />
        <StatusBadge 
          icon={Settings} 
          label="AUTO" 
          status={schedRunning ? true : false} 
          textOverride={schedRunning ? `ON /${schedInterval}s` : "OFF"}
        />
      </div>
    </div>
  );
}

function StatusBadge({ icon: Icon, label, status, textOverride }) {
  const color = status === null ? "var(--brand-gold)" : status ? "var(--brand-lime)" : "var(--brand-coral)";
  const text = textOverride || (status === null ? "..." : status ? "ONLINE" : "ERR");
  
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--text-muted)" }}>
      <div style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: color, boxShadow: `0 0 8px ${color}` }} />
      <span className="mono-text" style={{ fontSize: "11px" }}>{label} <span style={{ color }}>{text}</span></span>
    </div>
  );
}
