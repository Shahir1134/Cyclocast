import React, { useState } from 'react';
import { Search, Wind, Gauge, Calendar, Navigation, MapPin } from 'lucide-react';
import { getCategoryColor } from '../utils/constants';

export default function Sidebar({
  cyclones,
  selectedCycloneId,
  onSelectCyclone,
  isReplayMode,
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterBasin, setFilterBasin] = useState('all');

  const filtered = cyclones.filter((c) => {
    // Mode-based filtering
    if (isReplayMode && c.status !== 'historical') return false;
    if (!isReplayMode && c.status !== 'active') return false;

    // Search filter
    const matchesSearch = c.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          c.basin.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          c.category.toLowerCase().includes(searchTerm.toLowerCase());

    // Basin filter
    const matchesBasin = filterBasin === 'all' || c.basin.toLowerCase().includes(filterBasin.toLowerCase());

    return matchesSearch && matchesBasin;
  });

  return (
    <aside className="w-72 bg-gray-950 border-r border-gray-800 flex flex-col h-full flex-shrink-0 z-20 select-none">
      {/* Search Header */}
      <div className="p-3 border-b border-gray-800 space-y-2">
        <div className="relative">
          <Search className="w-4 h-4 text-gray-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder={isReplayMode ? "Search historical cyclones..." : "Search active storms..."}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
          />
        </div>

        {/* Basin filter tabs */}
        <div className="flex gap-1 text-[11px]">
          {['all', 'Bay of Bengal', 'Arabian Sea'].map((b) => (
            <button
              key={b}
              onClick={() => setFilterBasin(b)}
              className={`px-2 py-1 rounded-md transition-all capitalize flex-1 text-center font-medium ${
                filterBasin === b
                  ? 'bg-gray-800 text-white border border-gray-700'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {b === 'all' ? 'All Basins' : b === 'Bay of Bengal' ? 'BoB' : 'Arabian'}
            </button>
          ))}
        </div>
      </div>

      {/* Cyclone List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {filtered.length === 0 ? (
          <div className="p-6 text-center text-gray-500 text-xs">
            No storms match current criteria.
          </div>
        ) : (
          filtered.map((c) => {
            const isSelected = c.id === selectedCycloneId;
            const catColor = getCategoryColor(c.category);

            return (
              <div
                key={c.id}
                onClick={() => onSelectCyclone(c.id)}
                className={`p-3 rounded-xl border cursor-pointer transition-all ${
                  isSelected
                    ? 'bg-gradient-to-r from-blue-950/40 to-gray-900 border-blue-500 shadow-md shadow-blue-500/10'
                    : 'bg-gray-900/60 border-gray-800 hover:border-gray-700 hover:bg-gray-900'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-semibold text-white tracking-wide">
                      {c.name}
                    </h3>
                    <p className="text-[11px] text-gray-400 flex items-center gap-1">
                      <MapPin className="w-3 h-3 text-cyan-400" />
                      {c.basin} • {c.season}
                    </p>
                  </div>
                  <span
                    className="text-[10px] font-bold px-2 py-0.5 rounded-md border"
                    style={{
                      color: catColor,
                      borderColor: `${catColor}40`,
                      backgroundColor: `${catColor}15`,
                    }}
                  >
                    {c.category}
                  </span>
                </div>

                <div className="mt-2.5 pt-2 border-t border-gray-800/80 grid grid-cols-2 gap-2 text-[11px] text-gray-300">
                  <div className="flex items-center gap-1.5">
                    <Wind className="w-3.5 h-3.5 text-blue-400" />
                    <span>{Math.round(c.max_wind_kt)} kt ({Math.round(c.max_wind_kt * 1.852)} km/h)</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Gauge className="w-3.5 h-3.5 text-amber-400" />
                    <span>{Math.round(c.min_pressure_hpa)} hPa</span>
                  </div>
                </div>

                {c.movement && (
                  <div className="mt-1.5 text-[10px] text-cyan-300/80 flex items-center gap-1">
                    <Navigation className="w-3 h-3 transform rotate-45" />
                    <span>{c.movement}</span>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Footer Info */}
      <div className="p-2.5 border-t border-gray-800 bg-gray-950/80 text-[10px] text-gray-400 flex justify-between items-center">
        <span>Dataset: IMD / IBTrACS Best-Track</span>
        <span className="text-emerald-400 font-mono">LIVE API</span>
      </div>
    </aside>
  );
}
