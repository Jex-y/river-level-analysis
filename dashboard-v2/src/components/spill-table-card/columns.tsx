import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { type Status, StatusDot } from '@/components/ui/status-color-dot';
import { formatDate, formatDuration, statusFn } from './lib';
import { TooltipExplanation, SortingButton } from './sub-components';
import type { SpillEvent } from '@/types';
import type { ColumnDef } from '@tanstack/react-table';
import { Check, X as Cross } from 'lucide-react';

export const columns: ColumnDef<SpillEvent>[] = [
  {
    id: 'siteName',
    accessorKey: 'metadata.site_name',
    cell: ({ getValue }) => <span
      className='max-md:w-32 text-nowrap truncate inline-block'
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
      return nearby ? (
        <Check />
      ) : (
        <Cross />
      );
    },
  },
  {
    id: 'eventStart',
    accessorKey: 'event_start',
    cell: ({ getValue }) => <span>{formatDate(getValue() as Date)}</span>,
    sortingFn: 'datetime',
    header: ({ column }) => (
      <SortingButton column={column} label="Start Date" />
    ),
  },
  {
    id: 'eventEnd',
    accessorKey: 'event_end',
    cell: ({ getValue }) => <span>{formatDate(getValue() as Date)}</span>,
    sortingFn: 'datetime',
    header: ({ column }) => <SortingButton column={column} label="End Date" />,
  },
  {
    id: 'duration',
    // accessorFn: (row) => row.event_end.getTime() - row.event_start.getTime(),
    accessorKey: 'event_duration_mins',
    cell: ({ getValue }) => <span>{formatDuration(getValue() as number)}</span>,
    header: 'Duration',
  },
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
        <div className='flex items-center justify-center'>
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
