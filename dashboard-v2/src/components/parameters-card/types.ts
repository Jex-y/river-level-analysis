import type { Status } from '@/components/ui/status-color-dot';
import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

export type Parameters = {
	temperature: number;
	temperatureApparent: number;
	// uvIndex: number;
	visibility: number;
	windGust: number;
	windSpeed: number;
	riverLevel: number;
};

export type ParameterInfo = {
	label: string;
	defaultValue?: number;
	formatFn: (value: number) => ReactNode;
	statusFn: (value: number) => Status;
	icon: LucideIcon;
};
