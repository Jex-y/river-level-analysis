import { CircleAlert, CircleCheck, CircleX } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ParameterStatus, type Parameters, parameterInfo } from '@/lib/parametersConfig';
import { cn } from '@/lib/utils';
import { currentConditionsStore, currentWeatherForecastStore } from '@/store';
import { useStore } from '@nanostores/react';
import { type FC, useEffect, useState } from 'react';
import { Skeleton } from './ui/skeleton';



export type ParameterProps = {
  name: keyof Parameters;
};

const status_color = {
  Good: 'text-green-500',
  Warning: 'text-yellow-500',
  Bad: 'text-red-500',
};

type StatusIconProps = {
  status: ParameterStatus;
  className?: string;
};
const StatusIcon: FC<StatusIconProps> = ({ status, className }) => {
  switch (status) {
    case ParameterStatus.Good:
      return <CircleCheck className={cn(className, status_color.Good)} />;
    case ParameterStatus.Warning:
      return <CircleAlert className={cn(className, status_color.Warning)} />;
    case ParameterStatus.Bad:
      return <CircleX className={cn(className, status_color.Bad)} />;
    case ParameterStatus.Unknown:
      return <Skeleton className={cn(className, 'rounded-full')} />;
  }
};

export const Parameter: FC<ParameterProps> = ({
  name
}) => {
  const { label, formatFn, statusFn, defaultValue, icon: Icon } = parameterInfo[name];
  const [status, setStatus] = useState<ParameterStatus>(ParameterStatus.Unknown);
  const value = useStore(currentConditionsStore)?.[name] || defaultValue;

  useEffect(() => {
    if (value !== undefined) {
      setStatus(statusFn(value));
    }
  }, [value, statusFn]);

  return (
    <Card className="h-full p-2">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 p-2 px-2">
        <CardTitle className="text-xs lg:text-lg text-nowrap font-light">{label}</CardTitle>
        <Icon />
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <div className='flex flex-row items-center justify-between'>
          <div className="text-sm lg:text-lg font-semibold text-nowrap">
            {value !== undefined ? formatFn(value) : 'N/A'}
          </div>
          <StatusIcon status={status} className="h-6 w-6 lg:h-8 lg:w-8" />
        </div>
      </CardContent>
    </Card>
  );
};