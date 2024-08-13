"use client";

import * as config from './config';
import type { RiverLevel } from '@/types';

import {
  ChartContainer,
} from '@/components/ui/chart';

import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from 'recharts';

export function LevelChart({ data }: { data: RiverLevel[] }) {
  return (<ChartContainer
    config={config.chartConfig}
    className="aspect-auto h-[16rem] w-full"
  >
    <AreaChart data={data}>
      <CartesianGrid vertical={false} />
      <XAxis
        dataKey="timestamp"
        tickLine={true}
        axisLine={false}
        tickMargin={4}
        tickFormatter={(datetime) =>
          new Date(datetime / 1000).toLocaleTimeString('en-GB', {
            weekday: 'short',
            hour: '2-digit',
            minute: '2-digit',
          })
        }
        scale="time"
        type="number"
        domain={['auto', 'auto']}
      />
      <YAxis
        tickLine={false}
        axisLine={false}
        tickFormatter={(value) => value.toFixed(2)}
        unit={' m'}
        domain={[
          (dataMin: number) => Math.min(dataMin - 0.01, config.TYPICAL_LOW),
          (dataMax: number) => Math.max(dataMax + 0.01, config.TYPICAL_HIGH),
        ]}
      />
      {/* TODO: reimplement tooltip */}
      {/* <ChartTooltip
        cursor={true}
        content={
          <ChartTooltipContent
            indicator="dot"
            labelFormatter={(label) => {
              return new Date(
                payload[0].payload.timestamp / 1000
              ).toLocaleTimeString('en-GB', {
                weekday: 'short',
                hour: '2-digit',
                minute: '2-digit',
              });
            }}
          />
        }
      /> */}
      <defs>
        <linearGradient id="fillObserved" x1="0" y1="0" x2="0" y2="1">
          <stop
            offset="5%"
            stopColor="var(--color-observed)"
            stopOpacity={0.6}
          />
          <stop
            offset="95%"
            stopColor="var(--color-observed)"
            stopOpacity={0.1}
          />
        </linearGradient>
        <linearGradient id="fillPredicted" x1="0" y1="0" x2="0" y2="1">
          <stop
            offset="5%"
            stopColor="var(--color-predicted)"
            stopOpacity={0.6}
          />
          <stop
            offset="95%"
            stopColor="var(--color-predicted)"
            stopOpacity={0.1}
          />
        </linearGradient>
        <linearGradient id="fillStd" x1="0" y1="0" x2="0" y2="1">
          <stop
            offset="5%"
            stopColor="var(--color-std)"
            stopOpacity={0.6}
          />
          <stop
            offset="95%"
            stopColor="var(--color-std)"
            stopOpacity={0.1}
          />
        </linearGradient>
      </defs>
      <Area
        dataKey="observed"
        type="natural"
        fill="url(#fillObserved)"
        stroke="var(--color-observed)"
      />
      <Area
        dataKey="predicted"
        type="natural"
        fill="url(#fillPredicted)"
        stroke="var(--color-predicted)"
        strokeDasharray="5 5"
      />
      {/* <Area
              dataKey="std"
              type="natural"
              fill="url(#fillStd)"
              stroke="var(--color-std)"
              strokeDasharray={10}
            /> */}
    </AreaChart>
  </ChartContainer>);
}