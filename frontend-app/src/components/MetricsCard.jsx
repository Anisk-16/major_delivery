export function MetricsCard({ label, value, unit, sub, color="#00E5FF", icon: Icon, trend }) {
  return (
    <div className="glass-panel animate-fade-in" style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "8px", flex: 1, minWidth: "150px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--text-muted)", fontSize: "11px", letterSpacing: "1.5px", textTransform: "uppercase" }} className="mono-text">
        {Icon && <Icon size={14} color={color} />}
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: "4px" }}>
        <span className="mono-text" style={{ color: color, fontSize: "28px", fontWeight: 700, lineHeight: 1 }}>{value}</span>
        {unit && <span style={{ color: "var(--text-muted)", fontSize: "12px" }}>{unit}</span>}
      </div>
      {(sub || trend) && (
        <div style={{ display: "flex", flexDirection: "column", gap: "4px", marginTop: "auto" }}>
          {sub && <span style={{ color: "rgba(255,255,255,0.4)", fontSize: "11px" }}>{sub}</span>}
          {trend && <span className="mono-text" style={{ color: "var(--brand-lime)", fontSize: "11px" }}>{trend}</span>}
        </div>
      )}
    </div>
  );
}
