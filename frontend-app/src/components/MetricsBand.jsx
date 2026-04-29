import { MapPin, Clock, Fuel, PackageCheck, Truck, Zap, CloudFog, CloudLightning } from 'lucide-react';
import { MetricsCard } from './MetricsCard';

export function MetricsBand({ metrics }) {
  if (!metrics) return null;

  return (
    <div style={{ padding: "0 32px 24px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "16px" }}>
      <MetricsCard 
        label="TOTAL DISTANCE" 
        value={metrics.total_dist_km?.toFixed(2) || "0.00"} 
        unit="km" 
        color="var(--brand-cyan)" 
        icon={MapPin} 
        sub="Live route execution estimate"
      />
      <MetricsCard 
        label="TOTAL TIME" 
        value={metrics.total_time_min?.toFixed(0) || "0"} 
        unit="min" 
        color="var(--brand-gold)" 
        icon={Clock} 
      />
      <MetricsCard 
        label="FUEL EST." 
        value={metrics.total_fuel_L?.toFixed(3) || "0.000"} 
        unit="L" 
        color="var(--brand-coral)" 
        icon={Fuel} 
      />
      <MetricsCard 
        label="ON-TIME ETA" 
        value={metrics.on_time_pct || "0"} 
        unit="%" 
        color="var(--brand-lime)" 
        icon={PackageCheck} 
      />
      <MetricsCard 
        label="VEHICLES" 
        value={metrics.vehicles_used || "0"} 
        unit="" 
        color="var(--brand-purple)" 
        icon={Truck} 
      />
      <MetricsCard 
        label="CO2 SAVED" 
        value={metrics.co2_saved_kg?.toFixed(2) || "0.00"} 
        unit="kg" 
        color="var(--brand-lime)" 
        icon={CloudFog} 
        sub="vs flat baseline model"
      />
    </div>
  );
}
