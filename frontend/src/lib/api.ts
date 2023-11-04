import { DataPoint } from '@/types/entities';

export async function fetchData() {
	let datapoints: DataPoint[] = [];
	let last = Math.random() * 100;
	const NUM_POINTS = 100;
	for (let i = 0; i < NUM_POINTS; i++) {
		datapoints.push({
			timestamp: i,
			value: last + (Math.random() - 0.5) * 10,
		});
	}

	return datapoints;
}
