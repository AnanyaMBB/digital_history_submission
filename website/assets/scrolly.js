/* ============================================================
   THE BRITISH ACCESS NETWORK — scroll engine
   ============================================================ */
(function(){
  "use strict";
  document.documentElement.classList.add('js');
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- generic reveal observer ---------- */
  var revealObs = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if(e.isIntersecting){
        e.target.classList.add('in');
        if(e.target.hasAttribute('data-once')) revealObs.unobserve(e.target);
      }
    });
  },{threshold:0.16, rootMargin:'0px 0px -8% 0px'});

  function observeAll(sel){ document.querySelectorAll(sel).forEach(function(el){ revealObs.observe(el); }); }

  /* ---------- SVG line-draw setup ---------- */
  function prepDraw(){
    document.querySelectorAll('.draw').forEach(function(group){
      group.querySelectorAll('path,line,polyline,circle.ring').forEach(function(p){
        var len;
        try{ len = p.getTotalLength ? p.getTotalLength() : 1; }catch(err){ len = 1; }
        if(!len || isNaN(len)) len = 1;
        p.style.setProperty('--len', len);
      });
    });
  }

  /* ---------- counters ---------- */
  function animateCounter(el){
    var target = parseFloat(el.getAttribute('data-counter'));
    var dec = parseInt(el.getAttribute('data-dec')||'0',10);
    var dur = parseInt(el.getAttribute('data-dur')||'1500',10);
    var prefix = el.getAttribute('data-prefix')||'';
    var suffix = el.getAttribute('data-suffix')||'';
    if(reduce){ el.textContent = prefix + format(target,dec) + suffix; return; }
    var start = null;
    function format(v,d){ return d>0 ? v.toFixed(d).replace(/\B(?=(\d{3})+(?!\d))/g,',') : Math.round(v).toLocaleString('en-US'); }
    function step(ts){
      if(!start) start = ts;
      var prog = Math.min((ts-start)/dur,1);
      var eased = 1 - Math.pow(1-prog,3);
      el.textContent = prefix + format(target*eased,dec) + suffix;
      if(prog<1) requestAnimationFrame(step);
      else el.textContent = prefix + format(target,dec) + suffix;
    }
    requestAnimationFrame(step);
  }
  function format(v,d){ return d>0 ? v.toFixed(d) : Math.round(v).toLocaleString('en-US'); }
  var counterObs = new IntersectionObserver(function(entries){
    entries.forEach(function(e){ if(e.isIntersecting){ animateCounter(e.target); counterObs.unobserve(e.target); } });
  },{threshold:0.6});

  /* ---------- HERO ---------- */
  function initHero(){
    var stage = document.querySelector('.hero-stage');
    var q = document.querySelector('.hero-q');
    if(stage){ setTimeout(function(){ stage.classList.add('go'); }, reduce?0:350); }
    if(q){ setTimeout(function(){ q.classList.add('go'); }, reduce?0:2000); }
    // safety: never leave the hero blank if timing/visibility fails
    setTimeout(function(){ if(stage) stage.classList.add('go'); if(q) q.classList.add('go'); }, 4000);
    // backward gold line draws
    var hl = document.querySelector('.hero-line-wrap');
    if(hl){ setTimeout(function(){ hl.classList.add('in'); },2600); }
  }

  /* ---------- ACTORS NETWORK ---------- */
  function initActors(){
    var items = document.querySelectorAll('.actor-item');
    var nodes = document.querySelectorAll('.net-node[data-key]');
    var edges = document.querySelectorAll('.net-edge');
    function setActive(key){
      items.forEach(function(it){ it.classList.toggle('active', it.getAttribute('data-key')===key); });
      nodes.forEach(function(n){ n.classList.toggle('lit', n.getAttribute('data-key')===key || n.classList.contains('hub-core')); });
      edges.forEach(function(ed){ ed.classList.toggle('lit', ed.getAttribute('data-key')===key); });
    }
    items.forEach(function(it){
      it.addEventListener('mouseenter', function(){ setActive(it.getAttribute('data-key')); });
      it.addEventListener('click', function(){ setActive(it.getAttribute('data-key')); });
    });
    // progressive light-up as the network scrolls into view
    var stage = document.querySelector('.net-stage');
    if(stage){
      var lit=false;
      new IntersectionObserver(function(en){
        en.forEach(function(e){
          if(e.isIntersecting && !lit){
            lit=true;
            edges.forEach(function(ed,i){ setTimeout(function(){ ed.classList.add('lit'); },150+i*90); });
            nodes.forEach(function(n,i){ setTimeout(function(){ n.classList.add('lit'); },300+i*110); });
            setTimeout(function(){ if(items[0]) setActive(items[0].getAttribute('data-key')); },1400);
          }
        });
      },{threshold:0.4}).observe(stage);
    }
  }

  /* ---------- HORIZONTAL CRISIS TIMELINE ---------- */
  function initHTL(){
    var section = document.querySelector('.htl');
    var sticky = section && section.querySelector('.htl-sticky');
    var track = section && section.querySelector('.htl-track');
    var prog = section && section.querySelector('.htl-progress');
    if(!section||!track) return;
    function onScroll(){
      var rect = section.getBoundingClientRect();
      var total = section.offsetHeight - window.innerHeight;
      var passed = Math.min(Math.max(-rect.top,0), total);
      var p = total>0 ? passed/total : 0;
      var maxShift = track.scrollWidth - window.innerWidth;
      if(maxShift<0) maxShift = 0;
      track.style.transform = 'translateX(' + (-p*maxShift) + 'px)';
      if(prog) prog.style.width = (p*100).toFixed(2) + '%';
    }
    window.addEventListener('scroll', onScroll, {passive:true});
    window.addEventListener('resize', onScroll);
    onScroll();
  }

  /* ---------- LEAD-TIME BARS (sec 5) ---------- */
  function initLeadBars(){
    document.querySelectorAll('.lead-row').forEach(function(row){
      var bar = row.querySelector('.lead-bar');
      var sig = row.querySelector('.signal-dot');
      var rup = row.querySelector('.rupture-dot');
      var pct = parseFloat(row.getAttribute('data-pct'))||0; // bar width %
      new IntersectionObserver(function(en,obs){
        en.forEach(function(e){
          if(e.isIntersecting){
            if(bar) bar.style.transform = 'translateY(-50%) scaleX(' + (pct/100) + ')';
            if(sig) sig.style.left = '0%';
            if(rup) rup.style.left = pct + '%';
            obs.unobserve(row);
          }
        });
      },{threshold:0.5}).observe(row);
    });
  }

  /* ---------- TELEGRAPH PULSE (sec 6) ---------- */
  function initTelegraph(){
    var map = document.querySelector('.tele-map');
    if(!map) return;
    var ship = map.querySelector('.ship-dot');
    var pulse = map.querySelector('.cable-pulse');
    var shipPath = map.querySelector('#shipPath');
    var cablePath = map.querySelector('#cablePath');
    var played = false;
    function travel(dot, path, dur, cb){
      if(!dot||!path) { if(cb) cb(); return; }
      var len = path.getTotalLength();
      var t0=null;
      function f(ts){
        if(!t0) t0=ts;
        var p=Math.min((ts-t0)/dur,1);
        var pt = path.getPointAtLength(p*len);
        dot.setAttribute('transform','translate('+pt.x+','+pt.y+')');
        dot.style.opacity = p<1?1:0;
        if(p<1) requestAnimationFrame(f); else if(cb) cb();
      }
      dot.style.opacity=1;
      requestAnimationFrame(f);
    }
    new IntersectionObserver(function(en){
      en.forEach(function(e){
        if(e.isIntersecting && !played){
          played=true;
          // slow ship first
          travel(ship, shipPath, reduce?0:2600, function(){
            setTimeout(function(){ travel(pulse, cablePath, reduce?0:520); },420);
          });
        }
      });
    },{threshold:0.45}).observe(map);
  }

  /* ---------- IMM CHART (sec 8) ---------- */
  function initIMM(){
    document.querySelectorAll('.imm-chart.draw').forEach(function(c){
      revealObs.observe(c);
    });
  }

  /* ---------- CONTINUITY INDEX SIMULATOR (sec 10) ---------- */
  function initContinuity(){
    var root = document.getElementById('ci');
    if(!root) return;
    // criteria order: named, formal, mediated, routed, quant, direct
    var periods = [
      {id:'1847',  lab:'Victorian crises',           sub:'1847 – 1914',   conf:0.95, m:[1,0,1,1,1,1]},
      {id:'1906',  lab:'The bill market',            sub:'1906 baseline', conf:0.90, m:[1,1,1,0,1,0]},
      {id:'1931',  lab:'Sterling crisis',            sub:'1931',          conf:0.80, m:[1,1,1,1,0,1]},
      {id:'struct',lab:'Discount-market structure',  sub:'1830 – 1997',   conf:0.85, m:[1,1,1,1,0,0]},
      {id:'lifeboat',lab:'The lifeboat',             sub:'1973 – 1975',   conf:0.85, m:[1,1,1,1,1,1]},
      {id:'auth',  lab:'Authorised counterparties',  sub:'1976 – 1996',   conf:0.90, m:[1,1,1,0,1,0]},
      {id:'reform',lab:'The reform',                 sub:'1997',          conf:0.70, m:[0,0,0,0,0,0], reform:true}
    ];
    var enabled = [true,true,true,true,true,true];
    var rows = root.querySelectorAll('.ci-bar-row');
    var crits = root.querySelectorAll('.ci-crit');
    var readout = root.querySelector('#ci-readout');

    function render(animate){
      periods.forEach(function(pd,i){
        var row = rows[i]; if(!row) return;
        var score = 0;
        for(var k=0;k<6;k++){ if(enabled[k] && pd.m[k]) score++; }
        var fill = row.querySelector('.ci-barfill');
        var scoreEl = row.querySelector('.ci-score');
        var cells = row.querySelectorAll('.ci-cell');
        fill.style.width = (score/6*100) + '%';
        scoreEl.textContent = score;
        scoreEl.style.color = pd.reform ? 'var(--red)' : (score>=5?'var(--bank-green)':'var(--ink)');
        cells.forEach(function(cell,k){ cell.classList.toggle('on', !!(enabled[k] && pd.m[k])); });
      });
      // readout
      var activeCount = enabled.filter(Boolean).length;
      var span = '1847 to 1996';
      readout.innerHTML = 'Scoring <b>'+activeCount+' of 6</b> criteria. With all six on, the structure scores <b>4 to 6</b> across <b>'+span+'</b>, then drops to <b>0</b> at the 1997 reform. Turn criteria off to see which evidence each verdict rests on.';
    }
    crits.forEach(function(c,idx){
      c.addEventListener('click', function(){
        enabled[idx] = !enabled[idx];
        c.classList.toggle('on', enabled[idx]);
        render(true);
      });
    });
    var resetBtn = root.querySelector('.ci-reset');
    if(resetBtn) resetBtn.addEventListener('click', function(){
      enabled = [true,true,true,true,true,true];
      crits.forEach(function(c){ c.classList.add('on'); });
      render(true);
    });
    // initial paint when scrolled into view
    var painted=false;
    new IntersectionObserver(function(en){
      en.forEach(function(e){ if(e.isIntersecting && !painted){ painted=true; setTimeout(function(){ render(true); },200); } });
    },{threshold:0.3}).observe(root);
    // set state from markup
    crits.forEach(function(c){ c.classList.add('on'); });
  }

  /* ---------- SOURCE CARDS FLIP (sec 11) ---------- */
  function initSources(){
    document.querySelectorAll('.src-card').forEach(function(card){
      card.addEventListener('click', function(){ card.classList.toggle('flip'); });
    });
  }

  /* ---------- READING PROGRESS + CHAPTER TAG ---------- */
  function initProgress(){
    var bar = document.querySelector('.read-progress');
    var tag = document.querySelector('.chapter-tag');
    var sections = Array.prototype.slice.call(document.querySelectorAll('[data-chapter]'));
    function onScroll(){
      var h = document.documentElement;
      var sc = h.scrollTop || document.body.scrollTop;
      var max = h.scrollHeight - h.clientHeight;
      if(bar) bar.style.width = (max>0 ? sc/max*100 : 0) + '%';
      if(tag){
        var current=null;
        sections.forEach(function(s){ if(s.getBoundingClientRect().top < window.innerHeight*0.5) current=s; });
        if(current){ tag.textContent = current.getAttribute('data-chapter'); tag.classList.add('show'); }
        else tag.classList.remove('show');
      }
    }
    window.addEventListener('scroll', onScroll, {passive:true});
    onScroll();
  }

  /* ---------- INIT ---------- */
  function init(){
    prepDraw();
    observeAll('[data-animate]');
    observeAll('.draw');
    observeAll('.claim');
    observeAll('.firms-stage');
    document.querySelectorAll('[data-counter]').forEach(function(el){ counterObs.observe(el); });
    initHero();
    initActors();
    initHTL();
    initLeadBars();
    initTelegraph();
    initIMM();
    initContinuity();
    initSources();
    initProgress();
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
