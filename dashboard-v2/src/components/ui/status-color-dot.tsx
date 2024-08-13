import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';

export enum Status {
	Good = 0,
	Warning = 1,
	Bad = 2,
}

const status_color: Record<Status, string> = {
	[Status.Good]: 'bg-green-500',
	[Status.Warning]: 'bg-yellow-500',
	[Status.Bad]: 'bg-red-500',
};

export const SkeletonStatusDot = () => (
	<Skeleton className="h-3 w-3 rounded-full" />
);
export const StatusDot = ({ status }: { status: Status }) => (
	<div className={cn('h-3 w-3 rounded-full', status_color[status])} />
);
