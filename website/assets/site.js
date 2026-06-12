/* ============================================================
   THE BRITISH ACCESS NETWORK — nav router + crisis map + deck
   ============================================================ */
(function(){
  "use strict";
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------------- TAB ROUTER ---------------- */
  function initNav(){
    var tabs = document.querySelectorAll('.nav-tab');
    var pages = document.querySelectorAll('.page');
    var tabList = document.querySelector('.nav-tabs');
    var toggle = document.querySelector('.nav-toggle');

    function show(name){
      tabs.forEach(function(t){ t.classList.toggle('active', t.getAttribute('data-tab')===name); });
      pages.forEach(function(p){ p.classList.toggle('active', p.getAttribute('data-page')===name); });
      document.body.classList.toggle('story-active', name==='story');
      if(tabList) tabList.classList.remove('open');
      // scroll the active page to top, reset window
      window.scrollTo(0,0);
      // let scroll-driven sections recompute
      setTimeout(function(){ window.dispatchEvent(new Event('resize')); window.dispatchEvent(new Event('scroll')); }, 60);
      if(name==='presentation' && window.__deck) window.__deck.focus();
      if(history.replaceState) history.replaceState(null,'','#'+name);
    }
    tabs.forEach(function(t){
      t.addEventListener('click', function(){ show(t.getAttribute('data-tab')); });
    });
    if(toggle) toggle.addEventListener('click', function(){ tabList.classList.toggle('open'); });

    // deep-link support
    var hash = (location.hash||'').replace('#','');
    var valid = ['story','datasets','sources','methods','results','presentation'];
    show(valid.indexOf(hash)>=0 ? hash : 'story');
    window.__showTab = show;
  }

  /* ---------------- DATASET ACCORDION ---------------- */
  function initDatasets(){
    document.querySelectorAll('.ds-summary').forEach(function(s){
      s.addEventListener('click', function(){ s.closest('.ds-card').classList.toggle('open'); });
    });
  }

  /* ---------------- INTERACTIVE CRISIS MAP ---------------- */
  var CRISES = [
    {id:'1847', year:'1847', label:'Commercial crisis', type:'home', origin:'London',
     signal:'14 Jan 1847', rupture:'18 Oct 1847', lead:277,
     actors:['Bill brokers','Discount houses'],
     source:'BoE transaction ledgers · daily Bank Rate',
     note:'A long, visible build-up inside London finance before the public break.'},
    {id:'1857', year:'1857', label:'Imported panic', type:'external', origin:'New York',
     signal:'15 Oct 1857', rupture:'9 Nov 1857', lead:25,
     actors:['Discount houses','Overend, Gurney & Co.'],
     source:'BoE transaction ledgers · newspapers',
     note:'An imported shock crossed the Atlantic; the market-facing lead was short.'},
    {id:'1866', year:'1866', label:'The Overend panic', type:'home', origin:'London',
     signal:'4 Mar 1866', rupture:'11 May 1866', lead:288,
     actors:['Discount houses','Overend, Gurney (fails)'],
     source:'BoE ledgers · Flandreau & Ugolini 2011',
     note:'A long market build-up, then a sudden public rupture when Overend suspended.'},
    {id:'1890', year:'1890', label:'The Baring crisis', type:'home', origin:'London',
     signal:'26 Jun 1890', rupture:'15 Nov 1890', lead:142,
     actors:['Acceptance houses (Barings)','Clearing banks','The Lidderdale rescue'],
     source:'BoE Daily Account Books C1/38 · White 2016',
     note:'A home-grown shock handled through a recognised rescue circle.'},
    {id:'1907', year:'1907', label:'Imported panic', type:'external', origin:'New York',
     signal:'15 Aug 1907', rupture:'22 Oct 1907', lead:68,
     actors:['Discount houses'],
     source:'Daily Bank Rate · newspapers',
     note:'Four decades after the cable, the same kind of US-origin shock carried a longer lead, not a shorter one.'},
    {id:'1914', year:'1914', label:'The war shock', type:'external', origin:'Continent',
     signal:'30 Jul 1914', rupture:'31 Jul 1914', lead:1,
     actors:['Acceptance houses','Treasury','Joint-stock banks'],
     source:'BoE ledgers (reduced) · Roberts 2013',
     note:'War froze the system overnight; the lead almost vanished.'},
    {id:'1931', year:'1931', label:'Sterling crisis', type:'home', origin:'London',
     signal:'Sep 1931 · gold standard', rupture:'21 Sep 1931', lead:null,
     actors:['Big Five clearing banks','Seccombe Marshall & Campion','Discount houses'],
     source:'Römer 2025 reconstruction',
     note:'About £25m of Treasury bills bought through the discount houses; liquidity still ran through recognised market layers.'},
    {id:'1973', year:'1973–75', label:'The lifeboat', type:'home', origin:'London',
     signal:'Late 1973 · L&C fails', rupture:'Mar 1975 · peak', lead:null,
     actors:['Bank of England','Clearing banks','Control Committee'],
     source:'BoE Quarterly Bulletin 1978 Q2',
     note:'Support peaked at £1,285m across about 26 banks, vetted on a closed Control Committee. A selective network rescue, not an open facility.'},
    {id:'1997', year:'1997', label:'The reform', type:'reform', origin:'London',
     signal:'3 Mar 1997', rupture:'3 Mar 1997', lead:null,
     actors:['Wider eligible counterparty list'],
     source:'BoE institutional records',
     note:'The Bank ceased dealing exclusively with the LDMA and widened to a larger defined list. Access broadened to a bigger closed list, not the public.'}
  ];

  function initCrisisMap(){
    var root = document.getElementById('crisis-map');
    if(!root) return;
    var btns = root.querySelectorAll('.cm-pill');
    var maxLead = 288;

    function setRoute(c){
      var route = root.querySelector('#cm-route');
      var nyNode = root.querySelector('#cm-ny');
      var contNode = root.querySelector('#cm-cont');
      if(route) route.classList.toggle('on', c.origin==='New York');
      if(nyNode) nyNode.classList.toggle('on', c.origin==='New York');
      if(contNode) contNode.classList.toggle('on', c.origin==='Continent');
      // pulse London always; tint by type
      var lon = root.querySelector('#cm-london circle');
      if(lon) lon.setAttribute('fill', c.type==='external' ? 'var(--red)' : (c.type==='reform' ? 'var(--gold)' : 'var(--bank-green)'));
    }

    function select(c){
      btns.forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-id')===c.id); });
      // detail fields
      root.querySelector('#cm-year').textContent = c.year;
      root.querySelector('#cm-label').textContent = c.label;
      var typeEl = root.querySelector('#cm-type');
      typeEl.textContent = c.type==='home' ? 'Home-grown · London origin' : (c.type==='external' ? 'External shock · '+c.origin+' origin' : 'Institutional reform');
      typeEl.className = 'cm-type ' + c.type;
      root.querySelector('#cm-signal').textContent = c.signal;
      root.querySelector('#cm-rupture').textContent = c.rupture;
      root.querySelector('#cm-lead').textContent = (c.lead!=null) ? (c.lead + (c.lead===1?' day':' days')) : 'see note';
      root.querySelector('#cm-source').textContent = c.source;
      root.querySelector('#cm-note').textContent = c.note;
      // actors
      var ab = root.querySelector('#cm-actors'); ab.innerHTML='';
      c.actors.forEach(function(a){ var s=document.createElement('span'); s.className='cm-actor'; s.textContent=a; ab.appendChild(s); });
      // two clocks comparison
      var pct = (c.lead!=null) ? Math.max(c.lead/maxLead, 0.02)*100 : 100;
      var bar = root.querySelector('#cm-leadbar');
      var sigDot = root.querySelector('#cm-sig');
      var rupDot = root.querySelector('#cm-rup');
      bar.style.width = pct + '%';
      if(rupDot) rupDot.style.left = pct + '%';
      bar.classList.toggle('ext', c.type==='external');
      bar.classList.toggle('reform', c.type==='reform');
      var leadTxt = root.querySelector('#cm-leadtext');
      if(c.lead!=null){
        leadTxt.innerHTML = 'Market visibility led public information by <b>'+c.lead+(c.lead===1?' day':' days')+'</b>.';
      } else {
        leadTxt.innerHTML = 'A structural episode: access ran through recognised intermediaries rather than a single dated rupture.';
      }
      setRoute(c);
    }

    btns.forEach(function(b){
      b.addEventListener('click', function(){
        var c = CRISES.filter(function(x){ return x.id===b.getAttribute('data-id'); })[0];
        if(c) select(c);
      });
    });
    select(CRISES[2]); // default to 1866 (the long lead)
  }

  /* ---------------- PRESENTATION DECK ---------------- */
  function initDeck(){
    var deck = document.getElementById('deck');
    if(!deck) return;
    var slides = deck.querySelectorAll('.pslide');
    var prev = deck.querySelector('.deck-prev');
    var next = deck.querySelector('.deck-next');
    var counter = deck.querySelector('#deck-cur');
    var total = deck.querySelector('#deck-total');
    var progress = deck.querySelector('.deck-progress');
    var dotsWrap = deck.querySelector('.deck-dots');
    var notesBtn = deck.querySelector('.deck-notes-btn');
    var notesPanel = deck.querySelector('.deck-notes');
    var notesNo = deck.querySelector('#dn-no');
    var notesText = deck.querySelector('#dn-text');
    var notesRole = deck.querySelector('#dn-role');
    var n = slides.length, cur = 0;

    var NOTES = [
      {t:"Open with your name and the topic so it sticks. I am Ananya Besufekad, and this is The British Access Network. The question: from 1847 to 1997, did the Bank of England handle financial crises through an open public market, or a closed circle of recognised City firms? The visual: a crisis becomes public as a headline, but inside the City, stress was often visible earlier.", r:"Slide 1 · about 25 seconds"},
      {t:"State the question and the narrative. Did the Bank manage crises through an open public market, or a narrow recognised access network? If markets simply modernised, access should have opened. The argument here is different: prices got faster while access stayed mediated.", r:"Slide 2 · about 25 seconds"},
      {t:"Set up the debate. One tradition, Hopkins on gentlemanly capitalism, Amini and Toms on director networks, Gorton and Ordonez on crises as information events, Flandreau and Ugolini on the discount-house layer, sees finance shaped by elite access. The other, Hoag and the Richmond Fed, shows the 1866 cable made prices much faster, while Schneider and the Rothschild Archive warn against clean-rule or insider-profit stories. We test whether faster information also meant open access.", r:"Slide 3 · about 30 seconds"},
      {t:"Show the datasets. Nine sources, each answering one question. Ledgers show who reached the Bank. Bank Rate dates when stress became visible. Newspapers date the public rupture. Yale prices test whether access showed up as profit. The 1906 bills show the funnel. The Bank histories and the discount-house database carry the story into the twentieth century. No single source proves the case.", r:"Slide 4 · about 30 seconds"},
      {t:"Show the method pipeline. We turn scattered records into comparable clocks, networks, and scores. Sources are cleaned and classified, then turned into timing clocks, network and funnel measures, and a continuity score, before interpretation. This is what makes it digital history rather than a single archive reading.", r:"Slide 5 · about 25 seconds"},
      {t:"Result one, lead time. A market-facing signal appears before the public rupture in every crisis studied. The lead is longest when trouble grew inside London finance, 288 days in 1866 and 277 in 1847, and almost vanishes for the 1914 war shock at one day. Use the words early signal, market visibility, and lead time.", r:"Slide 6 · about 30 seconds"},
      {t:"Result two, the telegraph test. The 1866 cable cut the London to New York price lag toward zero. If the advantage were only fast news, the lead should have shrunk. Comparing two United States shocks, it grew, from 25 days in 1857 to 68 days in 1907. Faster prices did not open access.", r:"Slide 7 · about 30 seconds"},
      {t:"Result three, recurrence. The exact firms changed, but the access role persisted. A small set of recognised houses appears across crises, Union Discount in five periods, Overend Gurney until it fails in 1866, Seccombe Marshall and Campion bridging 1931 and the 1990s. The 1866 ledger chart shows who sat at the window.", r:"Slide 8 · about 30 seconds"},
      {t:"Result four and conclusion. The continuity score stays high across the crises, then the structure widens at the 1997 reform rather than simply opening. From bill brokers to authorised counterparties, crisis access stayed mediated through recognised actors until 1997. One honest caution: the price test shows no clear early repricing, so this is an access claim, not a profit claim.", r:"Slide 9 · about 35 seconds"},
      {t:"Close by repeating your name and the takeaway, since this is what students vote on. Crisis access at the Bank stayed mediated through recognised firms from 1847 to 1997. The club did not simply open, it was carried into formal rules. I am Ananya Besufekad, and this was The British Access Network. Thank you.", r:"Slide 10 · about 20 seconds"}
    ];

    if(total) total.textContent = n;
    // build dots
    for(var i=0;i<n;i++){
      (function(idx){
        var d = document.createElement('span'); d.className='deck-dot';
        d.addEventListener('click', function(){ go(idx); });
        dotsWrap.appendChild(d);
      })(i);
    }
    var dots = dotsWrap.querySelectorAll('.deck-dot');

    function go(i){
      cur = Math.max(0, Math.min(n-1, i));
      slides.forEach(function(s,idx){ s.classList.toggle('active', idx===cur); });
      dots.forEach(function(d,idx){ d.classList.toggle('on', idx===cur); });
      if(counter) counter.textContent = cur+1;
      if(progress) progress.style.width = ((cur+1)/n*100)+'%';
      if(prev) prev.disabled = cur===0;
      if(next) next.disabled = cur===n-1;
      if(notesNo) notesNo.textContent = (cur+1)+' / '+n;
      if(notesText) notesText.textContent = NOTES[cur].t;
      if(notesRole) notesRole.textContent = NOTES[cur].r;
    }
    if(prev) prev.addEventListener('click', function(){ go(cur-1); });
    if(next) next.addEventListener('click', function(){ go(cur+1); });
    if(notesBtn) notesBtn.addEventListener('click', function(){
      notesPanel.classList.toggle('open'); notesBtn.classList.toggle('on');
    });

    document.addEventListener('keydown', function(e){
      var onDeck = document.querySelector('.page[data-page="presentation"].active');
      if(!onDeck) return;
      if(e.key==='ArrowRight'||e.key==='PageDown'){ e.preventDefault(); go(cur+1); }
      else if(e.key==='ArrowLeft'||e.key==='PageUp'){ e.preventDefault(); go(cur-1); }
      else if(e.key==='Home'){ go(0); }
      else if(e.key==='End'){ go(n-1); }
    });

    window.__deck = { focus:function(){ /* hook for future */ }, go:go };
    go(0);
  }

  function init(){
    initNav();
    initDatasets();
    initCrisisMap();
    initDeck();
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
