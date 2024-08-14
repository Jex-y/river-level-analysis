import { type Status, StatusDot } from '@/components/ui/status-color-dot';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import type { SpillSite } from '@/types';
import type { ColumnDef } from '@tanstack/react-table';
import { Check, X as Cross } from 'lucide-react';
import { formatDate, formatDuration, statusFn } from './lib';
import { SortingButton, TooltipExplanation } from './sub-components';

export const columns: ColumnDef<SpillSite>[] = [
  {
    id: 'siteName',
    accessorKey: 'metadata.site_name',
    cell: ({ getValue }) => <span
      className='sm:w-48 text-nowrap truncate inline-block'
    >{getValue() as string}</span>,
    enableSorting: false,
    header: () => (
      <TooltipExplanation
        label="Site Name"
        explanation="The name of the site as given by Northumbrian Water."
      />
    ),
  },
  // {
  //   id: 'eventType',
  //   accessorKey: 'event_type',
  //   cell: ({ getValue }) => {
  //     const eventType = getValue() as 'spill' | 'monitor offline';
  //     return (
  //       <div className="flex items-center">
  //         {eventType === 'spill' ? (
  //           <Outflow className='text-muted' />
  //         ) : (
  //           <Offline />
  //         )}
  //       </div>
  //     );
  //   },
  //   enableSorting: false,
  //   header: 'Event Type',
  // },
  {
    id: 'nearby',
    accessorKey: 'metadata.nearby',
    header: () => (
      <TooltipExplanation
        label="Nearby"
        explanation="On or just upstream of the rowable section of river in Durham City."
      />
    ),
    cell: ({ getValue }) => {
      const nearby = getValue() as boolean;
      return (
        <div className="flex items-center justify-center">
          {nearby ? (
            <Check />
          ) : (
            <Cross />
          )}
        </div>
      );
    },

  },
  {
    id: 'recentSpill',
    accessorFn: (row) => {
      const spills = row.events.filter(
        (event) => event.event_type === 'spill'
      );

      if (spills.length === 0) {
        return "N/A";
      }

      return new Date(
        Math.max(
          ...spills
            .map((event) => event.event_end.getTime())
        ));
    },
    cell: ({ getValue }) => formatDate(getValue() as Date | string),
    sortingFn: 'datetime',
    header: ({ column }) => (
      <SortingButton column={column} label="Most Recent Spill" />
    ),
  },
  {
    id: 'totalSpillDuration',
    accessorFn: (row) => row.events.filter((event) => event.event_type === 'spill').reduce((acc, event) => acc + event.event_duration_mins, 0),
    cell: ({ getValue }) => formatDuration(getValue() as number),
    header: ({ column }) => <SortingButton column={column} label="Spill time past week" />,
  },
  // {
  //   id: 'totalOfflineTime',
  //   accessorFn: (row) => row.events.filter((event) => event.event_type === 'monitor offline').reduce((acc, event) => acc + event.event_duration_mins, 0),
  //   cell: ({ getValue }) => <span>{formatDuration(getValue() as number)}</span>,
  //   header: ({ column }) => <SortingButton column={column} label="Offline time past week" />,
  // },
  {
    id: 'status',
    accessorFn: (row) => statusFn(row),
    header: ({ column }) => <SortingButton column={column} label="Status" labelClassName='max-sm:hidden max-sm:sr-only' />,
    sortingFn: (rowA, rowB) =>
      (rowA.getValue('status') satisfies { status: Status }).status -
      (rowB.getValue('status') satisfies { status: Status }).status,
    cell: ({ getValue }) => {
      const { status, explanation } = getValue() as {
        status: Status;
        explanation: string;
      };
      return (
        <div className='text-center'>
          <TooltipProvider >
            <Tooltip>
              <TooltipTrigger>
                <StatusDot status={status} />
              </TooltipTrigger>
              <TooltipContent>
                <span>{explanation}</span>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      );
    },
  },
];
