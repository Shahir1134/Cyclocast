import React from 'react';
import { Wind, Gauge, ShieldAlert, AlertTriangle, Users, Anchor, ChevronRight, X } from 'lucide-react';
import { getCategoryColor, formatKnots } from '../utils/constants';

export default function SidePanel({
  cyclone,
  impactData,
  isOpen,
  onClose,
  onOpenImpactModal,
}) {
  if (!cyclone) return null;

  const catColor = getCategoryColor(cyclone.category);

  return (
    <div
      className={`absolute right-4 top-16 w-84 bg-gray-950/95 border border-gray-800 backdrop-blur-md rounded-2xl shadow-2xl p-4 z-20 flex flex-col gap-4 text-xs transition-all duration-300 ${
        isOpen ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0 pointer-events-none'
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-bold text-white tracking-wide">{cyclone.name}</h2>
            <span className="text-[10px] text-gray-400">({cyclone.season})</span>
          </div>
          <p className="text-[11px] text-gray-400">{cyclone.basin} • {cyclone.status.toUpperCase()}</p>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-md text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Primary Category Banner */}
      <div
        className="p-3 rounded-xl border flex items-center justify-between shadow-inner"
        style={{
          borderColor: `${catColor}60`,
          backgroundColor: `${catColor}15`,
        }}
      >
        <div>
          <span className="text-[10px] uppercase font-bold tracking-wider text-gray-300">IMD Classification</span>
          <h3 className="text-sm font-extrabold" style={{ color: catColor }}>
            {cyclone.category}
          </h3>
        </div>
        <ShieldAlert className="w-6 h-6" style={{ color: catColor }} />
      </div>

      {/* Plain Language Advisory */}
      {cyclone.advisory && (
        <div className="p-2.5 rounded-lg bg-blue-950/30 border border-blue-800/40 text-blue-200 text-[11px] leading-relaxed">
          <p className="font-semibold text-blue-300 flex items-center gap-1.5 mb-1">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            Operational Advisory
          </p>
          {cyclone.advisory}
        </div>
      )}

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 gap-2">
        <div className="p-2.5 rounded-xl bg-gray-900/80 border border-gray-800 flex flex-col">
          <span className="text-[10px] text-gray-400 flex items-center gap-1">
            <Wind className="w-3 h-3 text-cyan-400" />
            Max Sustained Wind
          </span>
          <span className="text-sm font-bold text-white mt-1">
            {Math.round(cyclone.max_wind_kt)} kt
          </span>
          <span className="text-[10px] text-gray-400">
            {Math.round(cyclone.max_wind_kt * 1.852)} km/h
          </span>
        </div>

        <div className="p-2.5 rounded-xl bg-gray-900/80 border border-gray-800 flex flex-col">
          <span className="text-[10px] text-gray-400 flex items-center gap-1">
            <Gauge className="w-3 h-3 text-amber-400" />
            Central Pressure
          </span>
          <span className="text-sm font-bold text-white mt-1">
            {Math.round(cyclone.min_pressure_hpa)} hPa
          </span>
          <span className="text-[10px] text-gray-400">Barometric minimum</span>
        </div>
      </div>

      {/* Coordinates & Movement */}
      <div className="p-2.5 rounded-xl bg-gray-900/60 border border-gray-800 flex justify-between items-center text-[11px] text-gray-300">
        <div>
          <span className="text-gray-400">Center: </span>
          <span className="font-mono text-cyan-400">{cyclone.current_lat}°N, {cyclone.current_lon}°E</span>
        </div>
        <div>
          <span className="text-gray-400">Speed: </span>
          <span>{cyclone.movement ? cyclone.movement.replace("Moving ", "") : "N/A"}</span>
        </div>
      </div>

      {/* Impact & Population Exposure Box */}
      {impactData && (
        <div className="p-3 rounded-xl bg-gray-900/90 border border-gray-800 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-gray-300 flex items-center gap-1.5">
              <Users className="w-3.5 h-3.5 text-orange-400" />
              Exposure Impact
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
              {impactData.coastal_alert_level.split(" ")[0]} Alert
            </span>
          </div>

          <div className="text-[11px] text-gray-300 flex justify-between items-baseline">
            <span>Pop. in cone:</span>
            <span className="font-bold text-white text-xs">
              {(impactData.total_exposed_population / 1000000).toFixed(2)} Million
            </span>
          </div>

          {impactData.vulnerable_districts?.length > 0 && (
            <div className="text-[10px] text-gray-400 flex items-center gap-1">
              <Anchor className="w-3 h-3 text-cyan-400" />
              Closest port: {impactData.vulnerable_districts[0].district_or_port} ({impactData.vulnerable_districts[0].distance_to_track_km} km)
            </div>
          )}

          <button
            onClick={onOpenImpactModal}
            className="mt-1 w-full py-1.5 rounded-lg bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 text-[11px] font-medium flex items-center justify-center gap-1 transition-all cursor-pointer"
          >
            <span>View Full Impact & Vulnerability</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}
