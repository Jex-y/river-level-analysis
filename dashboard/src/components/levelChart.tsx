import { levelForecastStore, levelObservationStore } from '@/store';
import { useStore } from '@nanostores/react';
import { type FC, useEffect, useState } from 'react';
import { useMediaQuery } from 'react-responsive';

import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import type { RiverLevel } from '@/lib/models';

const chartConfig = {
  observed: {
    label: "Observed",
    color: "hsl(var(--chart-1))",
  },
  predicted: {
    label: "Predicted",
    color: "hsl(var(--chart-1))",
  },
  ci: {
    label: "CI",
    color: "hsl(var(--chart-2))",
  },
} satisfies ChartConfig

export const LevelGraph: FC = () => {
  const [chartData, setChartData] = useState<RiverLevel[] | undefined>(undefined);
  const observedRiverLevel = useStore(levelObservationStore);
  const forecastRiverLevel = useStore(levelForecastStore);

  useEffect(() => {
    if (observedRiverLevel && forecastRiverLevel) {
      const firstPredicted = forecastRiverLevel[0];

      setChartData([
        ...observedRiverLevel,
        {
          timestamp: firstPredicted.timestamp,
          observed: firstPredicted.predicted,
        },
        ...forecastRiverLevel
      ]);
    }
  }, [observedRiverLevel, forecastRiverLevel]);


  const onMobile = useMediaQuery({ query: "(max-width: 768px)" });

  return (
    <Card className="h-full" >
      <CardHeader className="flex-row items-center justify-between p-4">
        <CardTitle>River Level</CardTitle>
        <CardDescription>
          Durham New Elvet Bridge
        </CardDescription>
      </CardHeader>
      <CardContent className='pl-0 pr-2'>
        <ChartContainer config={chartConfig} className="w-full">
          <AreaChart
            accessibilityLayer
            data={chartData}
          >
            <CartesianGrid vertical={true} />
            <XAxis
              dataKey="timestamp"
              tickLine={true}
              axisLine={true}
              tickMargin={8}
              tickFormatter={(datetime) =>
                new Date(datetime / 1000).toLocaleTimeString("en-GB", {
                  weekday: "short",
                  hour: "2-digit",
                  minute: "2-digit",
                })
              }
              scale='time'
              type='number'
              domain={['auto', 'auto']}
            />
            <YAxis
              tickLine={true}
              axisLine={true}
              tickMargin={onMobile ? 4 : 8}
              tickFormatter={(value) => value.toFixed(onMobile ? 1 : 2)}
              unit={' m'}
              domain={[
                (dataMin: number) => Math.max(dataMin - 0.01, 0),
                (dataMax: number) => dataMax + 0.01,
              ]}
            />
            <ChartTooltip
              cursor={true}
              content={<ChartTooltipContent indicator="dot" />}
            />
            <defs>
              <linearGradient id="fillObserved" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="var(--color-observed)"
                  stopOpacity={0.8}
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
                  stopOpacity={0.8}
                />
                <stop
                  offset="95%"
                  stopColor="var(--color-predicted)"
                  stopOpacity={0.1}
                />
              </linearGradient>
              <linearGradient id="fillCi" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="var(--color-ci)"
                  stopOpacity={0.8}
                />
                <stop
                  offset="95%"
                  stopColor="var(--color-ci)"
                  stopOpacity={0.1}
                />
              </linearGradient>
            </defs>
            <Area
              dataKey="observed"
              type="natural"
              fill="url(#fillObserved)"
              fillOpacity={0.4}
              stroke="var(--color-observed)"
            />
            <Area
              dataKey="predicted"
              type="natural"
              fill="url(#fillPredicted)"
              fillOpacity={0.4}
              stroke="var(--color-predicted)"
              strokeDasharray={10}
            />
            <Area
              dataKey="ci"
              type="natural"
              fill="url(#fillCi)"
              fillOpacity={0.4}
              stroke="var(--color-ci)"
              strokeDasharray={10}
            />
          </AreaChart>
        </ChartContainer>
      </CardContent>
      <CardFooter>
        <div className="text-muted-foreground text-xs">
          Predictions may be wildly inaccurate. Double check the observed data before boating!
        </div>
      </CardFooter>
    </Card >
  )
}
