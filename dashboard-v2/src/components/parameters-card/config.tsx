import { Status } from '@/components/ui/status-color-dot';
import { CloudFog, Sun, Thermometer, Waves, Wind } from 'lucide-react';
import type { ParameterInfo, Parameters } from './types';

// TODO: Implement sunrise and sunset

export const displayOrder: (keyof Parameters)[] = [
        'temperature',
        'temperatureApparent',
        // 'uvIndex',
        'visibility',
        'windGust',
        'windSpeed',
        'riverLevel',
];

type ValueWithUnitsProps = {
        value?: number;
        fractionDigits?: number;
        unit?: string;
};

function ValueWithUnits({
        value,
        fractionDigits = 1,
        unit,
}: ValueWithUnitsProps) {
        return (
                <div className="ml-auto flex justify-between items-baseline gap-1.5 font-mono font-medium tabular-nums text-foreground">
                        {value !== undefined ? value?.toFixed(fractionDigits) : '????'}
                        <span className="font-normal text-muted-foreground">{unit}</span>
                </div>
        );
}

export const parameterInfo: Record<keyof Parameters, ParameterInfo> = {
        temperature: {
                label: 'Temp',
                icon: Thermometer,
                formatFn: (value: number) => <ValueWithUnits value={value} unit="°C" />,
                statusFn: (value: number) => {
                        if (value < -5) {
                                return Status.Bad;
                        }

                        if (value < 5) {
                                return Status.Warning;
                        }

                        if (value > 25) {
                                return Status.Warning;
                        }

                        if (value > 30) {
                                return Status.Bad;
                        }

                        return Status.Good;
                },
        },
        temperatureApparent: {
                icon: Thermometer,
                label: 'Feels Like',
                formatFn: (value: number) => <ValueWithUnits value={value} unit="°C" />,
                statusFn: (value: number) => {
                        if (value < -5) {
                                return Status.Bad;
                        }

                        if (value < 5) {
                                return Status.Warning;
                        }

                        if (value > 25) {
                                return Status.Warning;
                        }

                        if (value > 35) {
                                return Status.Bad;
                        }

                        return Status.Good;
                },
        },
        // uvIndex: {
        // 	label: 'UV Index',
        // 	icon: Sun,
        // 	// Is null at night
        // 	defaultValue: 0,
        // 	formatFn: (value: number) => value.toFixed(0),
        // 	statusFn: (value: number) => {
        // 		if (value < 3) {
        // 			return Status.Good;
        // 		}

        // 		if (value < 8) {
        // 			return Status.Warning;
        // 		}

        // 		return Status.Bad;
        // 	},
        // },
        visibility: {
                label: 'Visibility',
                icon: CloudFog,
                formatFn: (value: number) => <ValueWithUnits value={value} unit="km" />,
                statusFn: (value: number) => {
                        if (value < 1) {
                                return Status.Bad;
                        }

                        if (value < 3) {
                                return Status.Warning;
                        }

                        return Status.Good;
                },
        },
        windGust: {
                label: 'Wind Gust',
                icon: Wind,
                formatFn: (value: number) => <ValueWithUnits value={value} unit="km/h" />,
                statusFn: (value: number) => {
                        const value_mph = value * 0.621371;

                        // 30 Miles per hour
                        if (value_mph < 30) {
                                return Status.Good;
                        }

                        // 45 Miles per hour
                        if (value_mph < 45) {
                                return Status.Warning;
                        }

                        return Status.Bad;
                },
        },
        windSpeed: {
                label: 'Wind Speed',
                icon: Wind,
                formatFn: (value: number) => <ValueWithUnits value={value} unit="km/h" />,
                statusFn: (value: number) => {
                        const value_mph = value * 0.621371;

                        if (value_mph < 15) {
                                return Status.Good;
                        }

                        if (value_mph < 25) {
                                return Status.Warning;
                        }

                        return Status.Bad;
                },
        },
        riverLevel: {
                label: 'River Level',
                icon: Waves,
                formatFn: (value: number) => (
                        <ValueWithUnits value={value} unit="m" fractionDigits={2} />
                ),
                statusFn: (value: number) => {
                        if (value < 0.4) {
                                return Status.Good;
                        }

                        if (value < 0.675) {
                                return Status.Warning;
                        }

                        return Status.Bad;
                },
        },
};

// TODO: Confirm these with Tomek
