export default function compareIntervals(windows: {resourceId:string;sceneId:string;window:string;status:string;startsAt?:string;endsAt?:string}[]): {overlaps:{left:number;right:number}[];unavailable:string[];unverified:string[]} {
  const overlaps:{left:number;right:number}[]=[], unavailable:string[]=[], unverified:string[]=[];
  const parse = (value: string | undefined): number | null => {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)) return null;
    const n=Date.parse(value); return Number.isFinite(n) && new Date(n).toISOString()===value ? n : null;
  };
  const intervals=windows.map((w,i)=>{
    if(w.status==="unavailable") unavailable.push(`${w.resourceId}:${w.sceneId}:${w.window}`);
    const start=parse(w.startsAt), end=parse(w.endsAt);
    if(start===null||end===null||end<=start) {unverified.push(`${i}:${w.resourceId}:${w.sceneId}:unresolved interval`);return null;}
    if(w.status==="unverified") unverified.push(`${i}:${w.resourceId}:${w.sceneId}:unverified availability`);
    return {start,end};
  });
  for(let i=0;i<windows.length;i++) for(let j=i+1;j<windows.length;j++) {
    const a=intervals[i],b=intervals[j];
    if(a&&b&&windows[i].resourceId===windows[j].resourceId&&windows[i].sceneId!==windows[j].sceneId&&a.start<b.end&&b.start<a.end) overlaps.push({left:i,right:j});
  }
  return {overlaps,unavailable,unverified};
}
