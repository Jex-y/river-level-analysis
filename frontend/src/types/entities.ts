import type { endpointsToOperations } from '../pages/api/[...entity].js';
import type { playgroundActions } from '../pages/playground/_actions.js';

export type EndpointsToOperations = typeof endpointsToOperations;
export type Endpoint = keyof EndpointsToOperations;

export type Products = Product[];
export interface Product {
	name: string;
	category: string;
	technology: string;
	id: number;
	description: string;
	price: string;
	discount: string;
}

export type Users = User[];
export interface User {
	id: number;
	name: string;
	avatar: string;
	email: string;
	biography: string;
	position: string;
	country: string;
	status: string;
}

export type PlaygroundAction = (typeof playgroundActions)[number];

export interface DataPoint {
	timestamp: number;
	value: number;
}

export interface ApiResult {
	historical: DataPoint[];
	predictions: DataPoint[];
	trend: 'up' | 'down' | 'flat';
}

export enum GoNoGo {
	Go,
	Warning,
	Nogo,
}
export interface Measurement {
	name: string;
	value: number;
	GoNoGo: GoNoGo;
	units: string;
}
