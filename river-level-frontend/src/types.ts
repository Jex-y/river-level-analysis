export enum MeasurementType {
  Actual = "actual",
  Forecast = "forecast",
}

export interface Measurement {
  timestamp: string;
  value: number;
  type: MeasurementType;
}