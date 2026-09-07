import React from 'react';
import {
  X,
  ShieldAlert,
  AlertTriangle,
  Users,
  Anchor,
  Compass,
  Building,
  CheckCircle2,
  Radio,
} from 'lucide-react';

export default function ImpactModal({
  isOpen,
  onClose,
  cyclone,
  impactData,
}) {
  if (!isOpen || !impactData) return null;

  const isExtreme = impactData.coastal_alert_level.toLowerCase().includes('extreme');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="bg-gray-950 border border-gray-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-5 border-b border-gray-800 flex items-center justify-between bg-gray-900/50">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl ${isExtreme ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'}`}>
              <ShieldAlert className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white tracking-wide">
                  Coastal Exposure & Impact Assessment
                </h2>
                <span className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${
                  isExtreme
                    ? 'bg-red-500/20 text-red-300 border-red-500/40'
                    : 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                }`}>
                  {impactData.coastal_alert_level}
                </span>
              </div>
              <p className="text-xs text-gray-400">
                Predicted Path Vulnerability for {cyclone?.name} ({cyclone?.basin})
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-xs">
          {/* Landfall Alert Banner */}
          {impactData.landfall_estimate && (
            <div className="p-4 rounded-xl bg-red-950/40 border border-red-800/50 text-red-200 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-bold text-sm text-red-300">Projected Landfall Zone</h4>
                <p className="text-xs leading-relaxed mt-0.5">
                  {impactData.landfall_estimate}
                </p>
              </div>
            </div>
          )}

          {/* Quick Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 flex flex-col justify-between">
              <div className="flex items-center justify-between text-gray-400">
                <span>Exposed Population</span>
                <Users className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="mt-2">
                <span className="text-2xl font-black text-white">
                  {(impactData.total_exposed_population / 1000000).toFixed(2)} M
                </span>
                <p className="text-[11px] text-gray-400 mt-0.5">In cone of uncertainty</p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 flex flex-col justify-between">
              <div className="flex items-center justify-between text-gray-400">
                <span>Closest Maritime Port</span>
                <Anchor className="w-4 h-4 text-amber-400" />
              </div>
              <div className="mt-2">
                <span className="text-base font-bold text-white">
                  {impactData.vulnerable_districts?.[0]?.district_or_port || 'Open Ocean'}
                </span>
                <p className="text-[11px] text-gray-400 mt-0.5">
                  {impactData.vulnerable_districts?.[0]?.distance_to_track_km
                    ? `${impactData.vulnerable_districts[0].distance_to_track_km} km from track`
                    : 'N/A'}
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 flex flex-col justify-between">
              <div className="flex items-center justify-between text-gray-400">
                <span>Vulnerable Zones</span>
                <Building className="w-4 h-4 text-purple-400" />
              </div>
              <div className="mt-2">
                <span className="text-2xl font-black text-white">
                  {impactData.vulnerable_districts?.length || 0}
                </span>
                <p className="text-[11px] text-gray-400 mt-0.5">Districts within 250 km radius</p>
              </div>
            </div>
          </div>

          {/* Infrastructure Alerts */}
          {impactData.infrastructure_alerts?.length > 0 && (
            <div>
              <h3 className="text-xs font-bold text-gray-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                Critical Infrastructure Alerts
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {impactData.infrastructure_alerts.map((alert, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-lg bg-gray-900/40 border border-gray-800 flex items-start gap-2"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 flex-shrink-0" />
                    <span className="text-[11px] text-gray-300 leading-normal">{alert}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Vulnerable Districts Table */}
          {impactData.vulnerable_districts?.length > 0 && (
            <div>
              <h3 className="text-xs font-bold text-gray-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Building className="w-3.5 h-3.5 text-cyan-400" />
                Vulnerable Coastal Districts & Ports
              </h3>
              <div className="border border-gray-800 rounded-xl overflow-hidden">
                <table className="w-full text-left text-xs text-gray-300">
                  <thead className="bg-gray-900/80 text-gray-400 text-[11px] border-b border-gray-800">
                    <tr>
                      <th className="py-2 px-3">District / Port</th>
                      <th className="py-2 px-3">State / Jurisdiction</th>
                      <th className="py-2 px-3 text-right">Distance to Track</th>
                      <th className="py-2 px-3 text-right">Estimated Population</th>
                      <th className="py-2 px-3 text-center">Vulnerability</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-900">
                    {impactData.vulnerable_districts.map((d, idx) => (
                      <tr key={idx} className="hover:bg-gray-900/40 transition-colors">
                        <td className="py-2.5 px-3 font-semibold text-white">
                          {d.district_or_port}
                        </td>
                        <td className="py-2.5 px-3 text-gray-400">
                          {d.state_or_country}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-cyan-400">
                          {d.distance_to_track_km} km
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono">
                          {d.estimated_population.toLocaleString()}
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          <span
                            className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                              d.risk_level === 'Extreme'
                                ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                                : d.risk_level === 'High'
                                ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                                : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                            }`}
                          >
                            {d.risk_level}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Operational Disaster Protocols */}
          <div className="p-4 rounded-xl bg-blue-950/20 border border-blue-900/30 space-y-2">
            <h4 className="font-bold text-xs text-blue-300 flex items-center gap-1.5">
              <Radio className="w-3.5 h-3.5 text-blue-400" />
              Standard IMD / NDMA Operational Protocols
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-[11px] text-gray-400">
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-blue-400 flex-shrink-0 mt-0.5" />
                <span>Total suspension of fishing and small boat operations across northern sector.</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-blue-400 flex-shrink-0 mt-0.5" />
                <span>Pre-positioning of NDRF/SDRF rescue teams in vulnerable coastal embankments.</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-blue-400 flex-shrink-0 mt-0.5" />
                <span>Maritime port signals 8 to 10 hoisted with cargo cranes locked in secure berths.</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800 bg-gray-900/40 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-white text-xs font-semibold transition-colors"
          >
            Close Assessment
          </button>
        </div>
      </div>
    </div>
  );
}
