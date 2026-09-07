import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceLine,
} from 'recharts';

export default function IntensityChart({ intensityData }) {
  if (!intensityData || !intensityData.data_points || intensityData.data_points.length === 0) {
    return (
      <div className="h-44 flex items-center justify-center text-gray-500 text-xs">
        No intensity timeline available.
      </div>
    );
  }

  const chartData = intensityData.data_points.map((pt) => ({
    label: pt.label,
    wind_kt: pt.wind_speed_kt,
    wind_kmh: pt.wind_speed_kmh,
    pressure: pt.pressure_hpa,
    category: pt.category,
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-900 border border-gray-700 p-2.5 rounded-lg shadow-xl text-xs text-white">
          <p className="font-bold text-cyan-400 mb-1">{label}</p>
          <p className="text-gray-300">Category: <span className="font-semibold text-amber-300">{data.category}</span></p>
          <p className="text-blue-400">Wind: <span className="font-bold">{data.wind_kt} kt</span> ({data.wind_kmh} km/h)</p>
          <p className="text-red-400">Pressure: <span className="font-bold">{data.pressure} hPa</span></p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-44 py-1">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 25, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="label" stroke="#6b7280" tick={{ fontSize: 10 }} />
          <YAxis
            yAxisId="wind"
            stroke="#3b82f6"
            tick={{ fontSize: 10 }}
            domain={['auto', 'auto']}
            label={{ value: 'Wind (kt)', angle: -90, position: 'insideLeft', fill: '#3b82f6', fontSize: 10 }}
          />
          <YAxis
            yAxisId="pressure"
            orientation="right"
            stroke="#ef4444"
            tick={{ fontSize: 10 }}
            domain={[920, 1020]}
            label={{ value: 'Pressure (hPa)', angle: 90, position: 'insideRight', fill: '#ef4444', fontSize: 10 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '4px' }} />

          {/* Tropical Cyclone Category Thresholds */}
          <ReferenceLine yAxisId="wind" y={64} stroke="#eab308" strokeDasharray="4 4" label={{ value: 'Severe (64kt)', fill: '#eab308', fontSize: 9 }} />
          <ReferenceLine yAxisId="wind" y={90} stroke="#dc2626" strokeDasharray="4 4" label={{ value: 'Very Severe (90kt)', fill: '#dc2626', fontSize: 9 }} />

          <Line
            yAxisId="wind"
            type="monotone"
            dataKey="wind_kt"
            name="Wind Speed (kt)"
            stroke="#3b82f6"
            strokeWidth={2.5}
            dot={{ r: 4, fill: '#3b82f6' }}
            activeDot={{ r: 6 }}
          />
          <Line
            yAxisId="pressure"
            type="monotone"
            dataKey="pressure"
            name="Pressure (hPa)"
            stroke="#ef4444"
            strokeWidth={2.5}
            strokeDasharray="3 3"
            dot={{ r: 4, fill: '#ef4444' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
