import { Status } from '@/components/ui/status-color-dot';
import type { SpillSite } from '@/types';

export const statusFn = (
	site: SpillSite
): { status: Status; explanation: string } => {
	const site_nearby = site.metadata.nearby;
	const has_spilled_in_last_week = site.events.some(
		(event) =>
			event.event_type === 'spill' &&
			new Date().getTime() - event.event_end.getTime() < 7 * 24 * 60 * 60 * 1000
	);
	const has_spilled_in_last_day = site.events.some(
		(event) =>
			event.event_type === 'spill' &&
			new Date().getTime() - event.event_end.getTime() < 24 * 60 * 60 * 1000
	);
	const has_been_offline_in_last_day = site.events.some(
		(event) =>
			event.event_type === 'monitor offline' &&
			new Date().getTime() - event.event_end.getTime() < 24 * 60 * 60 * 1000
	);

	if (site_nearby && has_spilled_in_last_day) {
		return {
			status: Status.Bad,
			explanation: 'This site is nearby and has spilled in the last 24 hours.',
		};
	}

	if (has_spilled_in_last_day) {
		return {
			status: Status.Warning,
			explanation:
				'This site has spilled in the last 24 hours, however it is not close to the rowable section of river in Durham City.',
		};
	}

	if (site_nearby && has_spilled_in_last_week) {
		return {
			status: Status.Warning,
			explanation: 'This site is nearby and has spilled in the last week.',
		};
	}

	if (has_spilled_in_last_week) {
		return {
			status: Status.Warning,
			explanation:
				'This site has spilled in the last week, however it is not close to the rowable section of river in Durham City.',
		};
	}

	if (site_nearby && has_been_offline_in_last_day) {
		return {
			status: Status.Warning,
			explanation:
				'This site is nearby and has been offline within the last 24 hours. This means that spills may not have been reported.',
		};
	}

	if (has_been_offline_in_last_day) {
		return {
			status: Status.Warning,
			explanation:
				'This site has been offline within the last 24 hours. This means that spills may not have been reported',
		};
	}

	return {
		status: Status.Good,
		explanation:
			'Site is online and has not recorded any spills in the last week.',
	};
};

export const formatDate = (date: Date | string) => {
	if (typeof date === 'string') {
		return date;
	}

	return date.toLocaleDateString('en-GB', {
		weekday: 'short',
		month: 'short',
		day: '2-digit',
		hour: '2-digit',
		minute: '2-digit',
	});
};

export const formatDuration = (duration: number) => {
	const days = Math.floor(duration / (60 * 24));
	const hours = Math.floor((duration % (60 * 24)) / 60);
	const minutes = duration % 60;

	if (days > 3) {
		return `${days} days`;
	}

	if (days > 1) {
		return `${days} days ${hours} hours`;
	}

	if (days > 0) {
		return `1 day ${hours} hours`;
	}

	if (hours > 0) {
		return `${hours} hours ${minutes} mins`;
	}

	return `${minutes} mins`;
};
