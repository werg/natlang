/** Measurement only. Private source references and reward logic never enter this container. */
import { chromium } from 'playwright-core';
let raw = '';
for await (const chunk of process.stdin) {
  raw += chunk;
  if (Buffer.byteLength(raw) > 262144) throw Error('request exceeds allocation');
}
const request = JSON.parse(raw);
if (typeof request.html !== 'string' || typeof request.token !== 'string') throw Error('invalid request');
const browser = await chromium.launch({headless:true, chromiumSandbox:true, args:['--disable-gpu']});
try {
  const pages = [];
  for (const width of [360, 768, 1280]) {
    const context = await browser.newContext({viewport:{width,height:900},deviceScaleFactor:1,
      javaScriptEnabled:false,serviceWorkers:'block',locale:'en-US',timezoneId:'UTC',reducedMotion:'reduce'});
    const blockedResources = [];
    await context.route('**/*', async route => {blockedResources.push(route.request().url().slice(0,200)); await route.abort();});
    const page = await context.newPage();
    await page.setContent(request.html, {waitUntil:'load',timeout:10000});
    await page.evaluate(() => document.fonts.ready);
    const measurements = await page.evaluate(() => {
      const norm = x => x.replace(/\s+/gu,' ').trim();
      const rect = r => ({x:r.x,y:r.y,width:r.width,height:r.height});
      const texts = [], anchors = [];
      const unsupportedPaint = [];
      for(const el of document.querySelectorAll('*')) {
        for(const pseudo of ['::before','::after']) {
          const style=getComputedStyle(el,pseudo);
          if(!['none','normal'].includes(style.content)) unsupportedPaint.push('generated-content');
        }
      }
      // Include pointer-transparent overlays in occlusion checks. This changes hit testing only.
      for(const el of document.querySelectorAll('*')) el.style.setProperty('pointer-events','auto','important');
      const walker = document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const node = walker.currentNode, parent = node.parentElement;
        const text = norm(node.textContent || '');
        if (!text || !parent || parent.closest('script,style,template,noscript')) continue;
        window.scrollTo(0,0);
        const range = document.createRange(); range.selectNodeContents(node);
        const boxes = [...range.getClientRects()].filter(r => r.width>0 && r.height>0);
        let visible = boxes.length>0, clip = false, opacity = 1, background = 'rgb(255, 255, 255)', foundBackground = false;
        for(let el=parent;el;el=el.parentElement) {
          const style=getComputedStyle(el), bounds=el.getBoundingClientRect();
          opacity *= Number(style.opacity);
          if(!foundBackground && !['rgba(0, 0, 0, 0)','transparent'].includes(style.backgroundColor)) {
            background=style.backgroundColor;foundBackground=true;
          }
          if(style.display==='none'||style.visibility!=='visible'||style.contentVisibility==='hidden') visible=false;
          if(style.filter!=='none'||style.clipPath!=='none'||style.maskImage!=='none'||style.clip!=='auto') visible=false;
          if(['hidden','clip','scroll','auto'].includes(style.overflowX))
            clip ||= boxes.some(r=>r.left<bounds.left-1||r.right>bounds.right+1);
          if(['hidden','clip','scroll','auto'].includes(style.overflowY))
            clip ||= boxes.some(r=>r.top<bounds.top-1||r.bottom>bounds.bottom+1);
        }
        const style=getComputedStyle(parent);
        const fill=style.webkitTextFillColor||style.color;
        // Fully transparent or same-color text cannot satisfy content preservation.
        visible &&= fill!=='rgba(0, 0, 0, 0)'&&fill!=='transparent'&&fill!==background;
        for(const box of boxes){
          if(box.right<=0||box.left>=innerWidth||box.bottom<=0){visible=false;continue;}
          window.scrollTo(0,Math.max(0,box.y+box.height/2-innerHeight/2));
          const hit=document.elementFromPoint(Math.max(0,Math.min(innerWidth-1,box.x+box.width/2)),
            Math.max(0,Math.min(innerHeight-1,box.y+box.height/2-scrollY)));
          if(!hit||(!parent.contains(hit)&&!hit.contains(parent))) visible=false;
        }
        texts.push({text,visible:visible&&opacity>=0.95,opacity,fontSize:parseFloat(style.fontSize),
          color:fill,background,clipped:clip,boxes:boxes.map(rect)});
      }
      window.scrollTo(0,0);
      for(const el of document.querySelectorAll('h1,h2,h3,h4,h5,h6,a,button,input,select,textarea')) {
        const style=getComputedStyle(el), box=el.getBoundingClientRect();
        anchors.push({tag:el.tagName.toLowerCase(),text:norm(el.textContent||''),
          type:el.getAttribute('type'),href:el.getAttribute('href'),name:el.getAttribute('name'),
          visible:box.width>0&&box.height>0&&style.visibility==='visible'&&style.display!=='none',box:rect(box)});
      }
      return {texts,anchors,unsupportedPaint,scrollWidth:document.documentElement.scrollWidth,
        scrollHeight:document.documentElement.scrollHeight,title:document.title};
    });
    pages.push({width,...measurements,blockedResources});
    await context.close();
  }
  const response = JSON.stringify({schema:'natlang.static-browser-measurements/1',token:request.token,pages});
  if(Buffer.byteLength(response)>1048576) throw Error('measurements exceed allocation');
  process.stdout.write(response+'\n');
} finally {await browser.close();}
