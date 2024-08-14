import * as logger from 'firebase-functions/logger';
import { onRequest } from 'firebase-functions/v2/https';

import { MongoClient } from 'mongodb';

const dbName = 'riverdata';
const collectionName = 'spills';
let _client: MongoClient | null = null;

const getClient = () => {
	if (!_client) {
		const url = process.env.DB_URI;
		if (!url) {
			throw new Error('DB_URI environment variable is required');
		}
		_client = new MongoClient(url);
	}

	return _client;
};

const fetch_spill_sites = async () => {
	const client = getClient();
	const db = client.db(dbName);
	const collection = db.collection(collectionName);

	logger.info('Fetching sewage leaks');

	const results = await collection
		.aggregate([
			{
				$match: {
					event_end: {
						$gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
					},
				},
			},
			{
				$addFields: {
					event_duration_mins: {
						$dateDiff: {
							startDate: '$event_start',
							endDate: '$event_end',
							unit: 'minute',
						},
					},
				},
			},
			{
				$unset: ['_id'],
			},
			{
				$match: {
					event_duration_mins: { $gte: 15 },
				},
			},
			{
				$sort: { event_end: -1 },
			},
			{
				$group: {
					_id: '$metadata.site_id',
					metadata: { $first: '$metadata' },
					events: {
						$push: {
							$unsetField: {
								field: 'metadata',
								input: '$$ROOT',
							},
						},
					},
				},
			},
		])
		.toArray();

	logger.info(`Found ${results.length} spill sites`);
	return results;
};

export const getSpillSites = onRequest(
	{
		region: 'europe-west2',
		cors: '*',
	},
	async (_req, res) => {
		res.set('Cache-Control', 'public, max-age=300, s-maxage=900');
		res.json(await fetch_spill_sites());
	}
);

// const results = await fetch_spill_sites();
// // Write to results.json
// import { writeFileSync } from 'fs';
// writeFileSync('results.json', JSON.stringify(results, null, 2));
