import { CloudFog, Sun, Thermometer, Waves, Wind } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export enum ParameterStatus {
	Good = 'Good',
	Warning = 'Warning',
	Bad = 'Bad',
}

// TODO: Implement sunrise and sunset
// Can we calculate this from latitude and datetime?

export type Parameters = {
	temperature: number;
	temperatureApparent: number;
	uvIndex: number;
	visibility: number;
	windGust: number;
	windSpeed: number;
	riverLevel: number;
};

export type ParameterInfo = {
	label: string;
	defaultValue?: number;
	formatFn: (value: number) => string;
	statusFn: (value: number) => ParameterStatus;
	icon: LucideIcon;
};

export const parameterInfo: Record<keyof Parameters, ParameterInfo> = {
	temperature: {
		label: 'Temp',
		icon: Thermometer,
		formatFn: (value: number) => `${value.toFixed(1)} °C`,
		statusFn: (value: number) => {
			if (value < -5) {
				return ParameterStatus.Bad;
			}

			if (value < 5) {
				return ParameterStatus.Warning;
			}

			if (value > 25) {
				return ParameterStatus.Warning;
			}

			if (value > 30) {
				return ParameterStatus.Bad;
			}

			return ParameterStatus.Good;
		},
	},
	temperatureApparent: {
		icon: Thermometer,
		label: 'Feels Like',
		formatFn: (value: number) => `${value.toFixed(1)} °C`,
		statusFn: (value: number) => {
			if (value < -5) {
				return ParameterStatus.Bad;
			}

			if (value < 5) {
				return ParameterStatus.Warning;
			}

			if (value > 25) {
				return ParameterStatus.Warning;
			}

			if (value > 35) {
				return ParameterStatus.Bad;
			}

			return ParameterStatus.Good;
		},
	},
	uvIndex: {
		label: 'UV Index',
		icon: Sun,
		// Is null at night
		defaultValue: 0,
		formatFn: (value: number) => value.toFixed(0),
		statusFn: (value: number) => {
			if (value < 3) {
				return ParameterStatus.Good;
			}

			if (value < 8) {
				return ParameterStatus.Warning;
			}

			return ParameterStatus.Bad;
		},
	},
	visibility: {
		label: 'Visibility',
		icon: CloudFog,
		formatFn: (value: number) => `${value.toFixed(1)} km`,
		statusFn: (value: number) => {
			if (value < 1) {
				return ParameterStatus.Bad;
			}

			if (value < 3) {
				return ParameterStatus.Warning;
			}

			return ParameterStatus.Good;
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
				return ParameterStatus.Good;
			}

			// 45 Miles per hour
			if (value_mph < 45) {
				return ParameterStatus.Warning;
			}

			return ParameterStatus.Bad;
		},
	},
	windSpeed: {
		label: 'Wind Speed',
		icon: Wind,
		formatFn: (value: number) => `${value.toFixed(1)} km/h`,
		statusFn: (value: number) => {
			const value_mph = value * 0.621371;

			if (value_mph < 15) {
				return ParameterStatus.Good;
			}

			if (value_mph < 25) {
				return ParameterStatus.Warning;
			}

			return ParameterStatus.Bad;
		},
	},
	riverLevel: {
		label: 'River Level',
		icon: Waves,
		formatFn: (value: number) => `${value.toFixed(2)} m`,
		statusFn: (value: number) => {
			if (value < 0.65) {
				return ParameterStatus.Good;
			}

			if (value < 0.675) {
				return ParameterStatus.Warning;
			}

			return ParameterStatus.Bad;
		},
	},
};

// TODO: Confirm these with Tomek
