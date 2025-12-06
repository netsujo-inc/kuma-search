/**
 * くま検知システム - ランディングページスクリプト
 */

document.addEventListener('DOMContentLoaded', function() {
  // モバイルナビゲーショントグル
  const navToggle = document.querySelector('.nav-toggle');
  const navLinks = document.querySelector('.nav-links');

  if (navToggle && navLinks) {
    navToggle.addEventListener('click', function() {
      navLinks.classList.toggle('active');
      navToggle.classList.toggle('active');
    });

    // リンクをクリックしたらメニューを閉じる
    navLinks.querySelectorAll('a').forEach(function(link) {
      link.addEventListener('click', function() {
        navLinks.classList.remove('active');
        navToggle.classList.remove('active');
      });
    });
  }

  // スムーススクロール
  document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const targetId = this.getAttribute('href');
      const target = document.querySelector(targetId);

      if (target) {
        const headerHeight = document.querySelector('.header').offsetHeight;
        const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - headerHeight;

        window.scrollTo({
          top: targetPosition,
          behavior: 'smooth'
        });
      }
    });
  });

  // スクロールアニメーション（Intersection Observer）
  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
  };

  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-in');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // アニメーション対象要素
  const animateElements = document.querySelectorAll(
    '.problem-card, .flow-item, .arch-box, .step, .tech-card, .pillar, .partner-card, .stat'
  );

  animateElements.forEach(function(el) {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
  });

  // アニメーションクラスのスタイル追加
  const style = document.createElement('style');
  style.textContent = `
    .animate-in {
      opacity: 1 !important;
      transform: translateY(0) !important;
    }
  `;
  document.head.appendChild(style);

  // ヘッダーのスクロール時のスタイル変更
  let lastScrollTop = 0;
  const header = document.querySelector('.header');

  window.addEventListener('scroll', function() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

    if (scrollTop > 100) {
      header.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)';
    } else {
      header.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.08)';
    }

    lastScrollTop = scrollTop;
  });

  // 統計数値のカウントアップアニメーション
  const statsObserver = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        const statNumbers = entry.target.querySelectorAll('.stat-number');
        statNumbers.forEach(function(stat) {
          const text = stat.textContent;
          // 数字を含む場合のみアニメーション
          const match = text.match(/^(\d+)/);
          if (match) {
            const targetNumber = parseInt(match[1], 10);
            const suffix = text.replace(/^\d+/, '');
            let current = 0;
            const increment = Math.ceil(targetNumber / 30);
            const timer = setInterval(function() {
              current += increment;
              if (current >= targetNumber) {
                current = targetNumber;
                clearInterval(timer);
              }
              stat.textContent = current + suffix;
            }, 50);
          }
        });
        statsObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  const heroStats = document.querySelector('.hero-stats');
  if (heroStats) {
    statsObserver.observe(heroStats);
  }

  // パートナーカードのホバーエフェクト強化
  document.querySelectorAll('.partner-card').forEach(function(card) {
    card.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-8px) scale(1.02)';
    });
    card.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0) scale(1)';
    });
  });
});
