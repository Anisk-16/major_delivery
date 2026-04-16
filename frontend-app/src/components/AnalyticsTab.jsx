import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, PieChart, Pie } from "recharts";

const TC = { Low:"var(--brand-lime)", Medium:"var(--brand-gold)", High:"var(--brand-coral)", Very_High:"#FF3CAC" };
const WC = ["#B0BEC5","#90A4AE","#64B5F6","#1E88E5","#78909C","#FF3CAC","#E0F7FA"];

export function AnalyticsTab({ rewardData, trafficData, weatherData }) {
  const chartStyle = {
    background: "rgba(0,0,0,0.3)",
    border: "1px solid var(--border-light)",
    borderRadius: "12px",
  };

  return (
    <div style={{ padding: "0 32px 32px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
      
      <div className="glass-panel" style={{ padding: "24px" }}>
        <h3 className="mono-text" style={{ fontSize: "14px", color: "var(--brand-cyan)", marginBottom: "20px", letterSpacing: "1px" }}>RL REWARD CURVE</h3>
        <div style={{ height: "250px", ...chartStyle }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rewardData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="step" stroke="var(--text-muted)" fontSize={12} />
              <YAxis stroke="var(--text-muted)" fontSize={12} />
              <Tooltip contentStyle={{ background: "var(--bg-deep)", border: "1px solid var(--border-light)", borderRadius: "8px" }} />
              <Line type="monotone" dataKey="reward" stroke="var(--brand-cyan)" strokeWidth={3} dot={{ r: 4, fill: "var(--brand-cyan)", strokeWidth: 0 }} activeDot={{ r: 6, fill: "var(--text-active)" }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: "24px" }}>
        <h3 className="mono-text" style={{ fontSize: "14px", color: "var(--brand-gold)", marginBottom: "20px", letterSpacing: "1px" }}>TRAFFIC DISTRIBUTION</h3>
        <div style={{ height: "250px", ...chartStyle }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={trafficData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }} barCategoryGap="40%">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={12} />
              <YAxis stroke="var(--text-muted)" fontSize={12} />
              <Tooltip cursor={{ fill: "rgba(255,255,255,0.05)" }} contentStyle={{ background: "var(--bg-deep)", border: "1px solid var(--border-light)", borderRadius: "8px" }} />
              <Bar dataKey="orders" radius={[6, 6, 0, 0]}>
                {trafficData.map((e, i) => <Cell key={i} fill={TC[e.name] || "var(--brand-purple)"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: "24px" }}>
        <h3 className="mono-text" style={{ fontSize: "14px", color: "var(--brand-purple)", marginBottom: "20px", letterSpacing: "1px" }}>WEATHER CONDITIONS</h3>
        <div style={{ height: "250px", ...chartStyle }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={weatherData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({name, percent}) => `${name} ${(percent*100).toFixed(0)}%`}>
                {weatherData.map((_, i) => <Cell key={i} fill={WC[i%WC.length]} />)}
              </Pie>
              <Tooltip contentStyle={{ background: "var(--bg-deep)", border: "1px solid var(--border-light)", borderRadius: "8px" }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: "24px" }}>
        <h3 className="mono-text" style={{ fontSize: "14px", color: "var(--brand-lime)", marginBottom: "20px", letterSpacing: "1px" }}>HYBRID PERFORMANCE (BENCHMARK)</h3>
        <div style={{ ...chartStyle, padding: "16px", height: "250px", overflow: "auto" }}>
          <table>
            <thead>
              <tr><th>Scenario</th><th>Distance</th><th>Time</th><th>Fuel</th></tr>
            </thead>
            <tbody className="mono-text">
              <tr style={{ color: "var(--text-muted)" }}><td>OR-Tools Baseline</td><td>43.3 km</td><td>104 min</td><td>5.2 L</td></tr>
              <tr style={{ color: "var(--brand-lime)" }}><td>Hybrid (RL Warm)</td><td>40.8 km</td><td>98 min</td><td>4.9 L</td></tr>
              <tr style={{ color: "var(--brand-coral)" }}><td>Traffic Event</td><td>45.0 km</td><td>108 min</td><td>5.4 L</td></tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
