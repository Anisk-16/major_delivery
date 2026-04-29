import { createElement } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Fuel,
  Gauge,
  ListChecks,
  Route,
  Timer,
  Trophy,
} from "lucide-react";

const asNumber = value => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const fmt = (value, digits = 2) => {
  const n = asNumber(value);
  return n === null ? "--" : n.toFixed(digits);
};

const statusColor = status => {
  if (status === "evaluated") return "var(--brand-lime)";
  if (status === "error") return "var(--brand-coral)";
  return "var(--brand-gold)";
};

export function ModelPerformanceTab({ rlMetrics }) {
  if (!rlMetrics) {
    return (
      <div style={{ padding: "0 32px 32px" }}>
        <div className="glass-panel mono-text" style={{ padding: "24px", color: "var(--text-muted)", fontSize: "13px" }}>
          Run the optimizer to calculate live PPO model performance metrics.
        </div>
      </div>
    );
  }

  const evaluated = rlMetrics.status === "evaluated";
  const cards = [
    {
      label: "PPO Reward Score",
      value: fmt(rlMetrics.reward_score, 3),
      sub: rlMetrics.reward_std != null
        ? `\u00b1${fmt(rlMetrics.reward_std, 3)} over ${rlMetrics.episodes_run ?? 1} episodes`
        : "negative route cost; higher is better",
      color: "var(--brand-cyan)",
      icon: Trophy,
    },
    {
      label: "Mean Reward / Order",
      value: fmt(rlMetrics.mean_reward_per_order, 3),
      sub: `${fmt(rlMetrics.orders_served, 1)} avg PPO decisions${rlMetrics.orders_served_std ? ` \u00b1${fmt(rlMetrics.orders_served_std, 1)}` : ""}`,
      color: "var(--brand-purple)",
      icon: Gauge,
    },
    {
      label: "On-Time Rate",
      value: `${fmt(rlMetrics.on_time_pct, 1)}%`,
      sub: rlMetrics.on_time_std != null
        ? `\u00b1${fmt(rlMetrics.on_time_std, 1)}% variability across episodes`
        : `${rlMetrics.on_time_count ?? 0}/${rlMetrics.orders_served ?? 0} served on time`,
      color: "var(--brand-lime)",
      icon: CheckCircle2,
    },
    {
      label: "Late Orders",
      value: fmt(rlMetrics.late_count, 1),
      sub: rlMetrics.late_std != null
        ? `\u00b1${fmt(rlMetrics.late_std, 1)} std — avg over ${rlMetrics.episodes_run ?? 1} episodes`
        : `${rlMetrics.orders_served ?? 0} orders evaluated by PPO`,
      color: (rlMetrics.late_count || 0) === 0 ? "var(--brand-lime)" : "var(--brand-coral)",
      icon: AlertTriangle,
    },
    {
      label: "RL Distance",
      value: `${fmt(rlMetrics.total_dist_km, 2)} km`,
      sub: "avg distance traveled by PPO policy",
      color: "var(--brand-cyan)",
      icon: Route,
    },
    {
      label: "RL Fuel",
      value: `${fmt(rlMetrics.total_fuel_L, 3)} L`,
      sub: "avg fuel used by PPO policy",
      color: "var(--brand-gold)",
      icon: Fuel,
    },
    {
      label: "Policy Inference",
      value: `${fmt(rlMetrics.inference_time_s, 4)}s`,
      sub: `${fmt(rlMetrics.avg_decision_time_ms, 3)} ms per decision`,
      color: "var(--brand-purple)",
      icon: Timer,
    },
    {
      label: "Batch Coverage",
      value: `${fmt(rlMetrics.coverage_pct, 1)}%`,
      sub: `${rlMetrics.evaluated_orders ?? 0}/${rlMetrics.requested_orders ?? 0} orders evaluated`,
      color: (rlMetrics.skipped_orders || 0) > 0 ? "var(--brand-gold)" : "var(--brand-lime)",
      icon: ListChecks,
    },
  ];

  const detailRows = [
    ["Status",             (rlMetrics.status || "unknown").toUpperCase()],
    ["Model Loaded",       rlMetrics.model_loaded ? "YES" : "NO"],
    ["Episodes Run",       rlMetrics.episodes_run ?? 1],
    ["Requested Orders",   rlMetrics.requested_orders ?? 0],
    ["Evaluated Orders",   rlMetrics.evaluated_orders ?? 0],
    ["Avg Orders Served",  rlMetrics.orders_served != null ? `${rlMetrics.orders_served} ± ${rlMetrics.orders_served_std ?? 0}` : "--"],
    ["Skipped Orders",     rlMetrics.skipped_orders ?? 0],
    ["Max Supported Orders", rlMetrics.max_supported_orders ?? "--"],
    ["Reward Std Dev",     rlMetrics.reward_std != null ? fmt(rlMetrics.reward_std, 3) : "--"],
    ["Action Count (avg)", rlMetrics.action_count ?? 0],
  ];
  const actions = rlMetrics.action_sequence || [];

  return (
    <div style={{ padding: "0 32px 32px", display: "grid", gap: "24px" }}>
      <div className="glass-panel" style={{ padding: "22px 24px", display: "grid", gap: "10px" }}>
        <div className="mono-text" style={{ display: "flex", alignItems: "center", gap: "10px", color: statusColor(rlMetrics.status), fontSize: "14px", letterSpacing: "1px" }}>
          <Activity size={16} />
          PPO POLICY PERFORMANCE
        </div>
        <div style={{ color: "var(--text-muted)", fontSize: "13px", lineHeight: 1.5 }}>
          {rlMetrics.reason || "Live PPO metrics from the latest optimizer run."}
        </div>
      </div>

      {evaluated && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "16px" }}>
          {cards.map(card => (
            <ScoreCard key={card.label} {...card} />
          ))}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(280px, 0.7fr)", gap: "24px" }}>
        <div className="glass-panel" style={{ padding: "24px" }}>
          <h3 className="mono-text" style={{ fontSize: "14px", color: "var(--brand-cyan)", marginBottom: "18px", letterSpacing: "1px" }}>
            LIVE PPO RUN DETAILS
          </h3>
          <table>
            <tbody className="mono-text">
              {detailRows.map(([label, value]) => (
                <tr key={label}>
                  <td style={{ color: "var(--text-muted)" }}>{label}</td>
                  <td style={{ color: label === "Status" ? statusColor(rlMetrics.status) : "var(--text-main)" }}>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="glass-panel" style={{ padding: "24px" }}>
          <h3 className="mono-text" style={{ fontSize: "14px", color: "var(--brand-gold)", marginBottom: "18px", letterSpacing: "1px" }}>
            PPO ACTION SAMPLE
          </h3>
          {actions.length ? (
            <div className="mono-text" style={{ display: "grid", gap: "8px", fontSize: "12px" }}>
              {actions.slice(0, 8).map(action => (
                <div key={action.step} style={{ display: "flex", justifyContent: "space-between", gap: "12px", color: "var(--text-muted)" }}>
                  <span>Step {action.step}</span>
                  <span style={{ color: "var(--brand-cyan)" }}>Order {action.selected_order_id ?? action.selected_order_index}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="mono-text" style={{ color: "var(--text-muted)", fontSize: "12px", lineHeight: 1.5 }}>
              No PPO actions were returned for this run.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ScoreCard({ label, value, sub, color, icon }) {
  return (
    <div className="glass-panel animate-fade-in" style={{ padding: "18px", display: "grid", gap: "10px" }}>
      <div className="mono-text" style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--text-muted)", fontSize: "11px", letterSpacing: "1px", textTransform: "uppercase" }}>
        {icon && createElement(icon, { size: 15, color })}
        {label}
      </div>
      <div className="mono-text" style={{ color, fontSize: "28px", fontWeight: 700, lineHeight: 1 }}>
        {value}
      </div>
      <div style={{ color: "rgba(255,255,255,0.45)", fontSize: "12px" }}>
        {sub}
      </div>
    </div>
  );
}
