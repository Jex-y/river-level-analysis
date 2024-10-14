import type { ChartConfig } from '@/components/ui/chart';

export const TYPICAL_LOW = 0.226;
// const TYPICAL_HIGH = 2.630;
export const TYPICAL_HIGH = 0.75;

export const chartConfig = {
        value: {
                label: 'Observed',
                color: 'hsl(var(--chart-1))',
        },
        mean: {
                label: 'Expected',
                color: 'hsl(var(--chart-2))',
        },
        std: {
                label: '1 Std Dev',
                color: 'hsl(var(--chart-2))',
        },
} satisfies ChartConfig;

export const referenceLines = [
        {
                label: 'DCR Limit',
                value: 0.65,
        },
];
