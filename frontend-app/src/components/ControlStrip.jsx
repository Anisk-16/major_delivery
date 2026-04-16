import { useState } from 'react';
import { Package, Route, Activity, BarChart2, Database } from 'lucide-react';
import { MetricsCard } from './MetricsCard';

export function ControlStrip({ 
  orders, nOrders, setNOrders, nVehicles, setNVehicles, capacity, setCapacity,
  loading, loadOrders, runOptimize 
}) {
  return (
    <div className="glass-panel" style={{ margin: "24px 32px", padding: "16px 24px", display: "flex", alignItems: "center", gap: "24px", flexWrap: "wrap", border: "1px solid var(--border-focus)" }}>
      <ControlInput label="ORDERS" value={nOrders} onChange={setNOrders} min={5} max={200} />
      <ControlInput label="VEHICLES" value={nVehicles} onChange={setNVehicles} min={1} max={10} />
      <ControlInput label="CAPACITY" value={capacity} onChange={setCapacity} min={1} max={50} />
      
      <div style={{ width: "1px", height: "30px", background: "var(--border-light)" }} />

      <button className="btn-primary" onClick={loadOrders} disabled={loading}>
        <Database size={16} />
        {loading ? "LOADING..." : "LOAD DATASET"}
      </button>

      <button className="btn-success" onClick={runOptimize} disabled={loading || !orders.length}>
        <Route size={16} />
        {loading ? "SOLVING..." : "RUN OPTIMIZER"}
      </button>

      <div style={{ marginLeft: "auto", fontSize: "12px", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "8px" }} className="mono-text animate-fade-in">
        <Package size={14} color="var(--brand-cyan)" />
        {orders.length} in buffer
      </div>
    </div>
  );
}

function ControlInput({ label, value, onChange, min, max }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
      <span className="mono-text" style={{ fontSize: "11px", color: "var(--text-muted)" }}>{label}</span>
      <input 
        type="number" 
        value={value} 
        onChange={e => onChange(+e.target.value)} 
        min={min} 
        max={max}
        className="mono-text"
        style={{ 
          background: "rgba(0,0,0,0.3)", border: "1px solid var(--border-light)", 
          borderRadius: "6px", color: "var(--text-active)", padding: "8px 12px", 
          width: "70px", outline: "none", textAlign: "center" 
        }} 
      />
    </div>
  );
}
