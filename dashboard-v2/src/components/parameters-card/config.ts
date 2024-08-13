import { Status } from '@/components/ui/status-color-dot';
import { CloudFog, Sun, Thermometer, Waves, Wind } from 'lucide-react';
import type {Parameters, ParameterInfo} from './types';

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

export const parameterInfo: Record<keyof Parameters, ParameterInfo> = {
	temperature: {
		label: 'Temp',
		icon: Thermometer,
		formatFn: (value: number) => `${value.toFixed(1)} °C`,
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
		formatFn: (value: number) => `${value.toFixed(1)} °C`,
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
		formatFn: (value: number) => `${value.toFixed(1)} km`,
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
		formatFn: (value: number) => `${value.toFixed(1)} km/h`,
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
		formatFn: (value: number) => `${value.toFixed(1)} km/h`,
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
		formatFn: (value: number) => `${value.toFixed(2)} m`,
		statusFn: (value: number) => {
			if (value < 0.65) {
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
