/** What the pages read from a failed API call (an axios error). */
export interface ApiError<Detail = string> {
  response?: { data?: { detail?: Detail } };
}
