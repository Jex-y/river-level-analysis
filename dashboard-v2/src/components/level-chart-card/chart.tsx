"use client";

import type { RiverLevel } from "@/types";
import * as config from "./config";

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";

const useWithNumberTimestamp = (data: RiverLevel[]) => {
  return data.map((d) => ({
    ...d,
    timestamp: new Date(d.timestamp).getTime(),
  }));
};

export function LevelChart({ data }: { data: RiverLevel[] }) {
  let dataWithNumberTimestamp = useWithNumberTimestamp(data);

  return (
    <ChartContainer
      config={config.chartConfig}
      className="aspect-auto h-[16rem] w-full"
    >
      <AreaChart data={dataWithNumberTimestamp}>
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey="timestamp"
          tickLine={true}
          axisLine={false}
          tickMargin={4}
          tickFormatter={(datetime) =>
            new Date(datetime / 1000).toLocaleTimeString("en-GB", {
              weekday: "short",
              hour: "2-digit",
              minute: "2-digit",
            })
          }
          scale="time"
          type="number"
          domain={["auto", "auto"]}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tickFormatter={(value) => value.toFixed(2)}
          unit={" m"}
          domain={[
            (dataMin: number) => Math.min(dataMin - 0.01, config.TYPICAL_LOW),
            (dataMax: number) => Math.max(dataMax + 0.01, config.TYPICAL_HIGH),
          ]}
        />
        {config.referenceLines.map(({ label, value }) => (
          <ReferenceLine
            key={label}
            y={value}
            // stroke="var(--color-reference)"
            // strokeDasharray="5 5"
            label={{ value: label, position: "left" }}
          />
        ))}
        <ChartTooltip
          content={
            <ChartTooltipContent
              labelFormatter={(_value, payload) => {
                const date = new Date(payload[0].payload.timestamp / 1000);

                return new Date(date).toLocaleDateString("en-GB", {
                  weekday: "short",
                  day: "numeric",
                  month: "short",
                  hour: "2-digit",
                  minute: "2-digit",
                });
              }}
              formatter={(value, name) => (
                <>
                  <div
                    className="h-2.5 w-2.5 shrink-0 rounded-[2px] bg-[--color-bg]"
                    style={
                      {
                        "--color-bg": `var(--color-${name})`,
                      } as React.CSSProperties
                    }
                  />
                  {config.chartConfig[name as keyof typeof config.chartConfig]
                    ?.label || name}
                  <div className="ml-auto flex items-baseline gap-1 font-mono font-medium tabular-nums text-foreground">
                    {(value as number).toFixed(2)}
                    <span className="font-normal text-muted-foreground">m</span>
                  </div>
                </>
              )}
            />
          }
          cursor={false}
          defaultIndex={1}
        />
        <defs>
          <linearGradient id="fillValue" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="5%"
              stopColor="var(--color-value)"
              stopOpacity={0.6}
            />
            <stop
              offset="95%"
              stopColor="var(--color-value)"
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
            <stop offset="5%" stopColor="var(--color-std)" stopOpacity={0.6} />
            <stop offset="95%" stopColor="var(--color-std)" stopOpacity={0.1} />
          </linearGradient>
        </defs>
        <Area
          dataKey="value"
          type="natural"
          fill="url(#fillValue)"
          stroke="var(--color-value)"
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
    </ChartContainer>
  );
}
