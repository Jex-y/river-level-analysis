import type { ChartConfig } from '@/components/ui/chart';

export const TYPICAL_LOW = 0.226;
// const TYPICAL_HIGH = 2.630;
export const TYPICAL_HIGH = 0.75;

export const chartConfig = {
  observed: {
    label: 'Observed',
    color: 'hsl(var(--chart-1))',
  },
  predicted: {
    label: 'Predicted',
    color: 'hsl(var(--chart-4))',
  },
  std: {
    label: '1 Std Dev',
    color: 'hsl(var(--chart-2))',
  },
} satisfies ChartConfig;