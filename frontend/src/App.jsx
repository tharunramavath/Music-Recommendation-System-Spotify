import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Search, Music, Heart, SkipForward, Play, LayoutDashboard, Compass, Radio, Mic2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const BASE_URL = 'http://localhost:8000';

const App = () => {
  const [user_id, setUserId] = useState(123); 
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [onboarded, setOnboarded] = useState(false);
  const [topGenres, setTopGenres] = useState([]);
  const [selectedGenres, setSelectedGenres] = useState([]);

  useEffect(() => {
    fetchGenres();
  }, []);

  const fetchGenres = async () => {
    try {
      const res = await axios.get(`${BASE_URL}/genres`);
      setTopGenres(res.data.top_genres);
    } catch (e) { console.error('Failed to fetch genres', e); }
  };

  const startOnboarding = async () => {
    setLoading(true);
    try {
      await axios.post(`${BASE_URL}/onboarding?user_id=${user_id}`, selectedGenres);
      setOnboarded(true);
      fetchRecommendations();
    } catch (e) {
      alert('Onboarding failed. Is the API running?');
    } finally { setLoading(false); }
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${BASE_URL}/recommend/${user_id}?k=24`);
      setRecommendations(res.data.recommendations);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  const handleSearch = async (e) => {
    if (e.key === 'Enter') {
      setLoading(true);
      try {
        const res = await axios.get(`${BASE_URL}/search?query=${searchQuery}`);
        setRecommendations(res.data.results);
      } catch (e) { console.error(e); }
      finally { setLoading(false); }
    }
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="flex items-center gap-3 mb-10" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="w-10 h-10 bg-spotify rounded-full flex items-center justify-center" style={{ width: '40px', height: '40px', background: 'var(--spotify)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Music size={20} color="black" />
          </div>
          <span className="font-bold text-2xl tracking-tight">Spotify Music Recommender</span>
        </div>
        
        <nav className="flex flex-col gap-6" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <a href="#" className="flex items-center gap-4 text-white font-semibold" style={{ display: 'flex', alignItems: 'center', gap: '16px', color: 'white', textDecoration: 'none' }}>
            <LayoutDashboard size={22} /> Dashboard
          </a>
          <a href="#" className="flex items-center gap-4 text-text-muted hover:text-white transition-all" style={{ display: 'flex', alignItems: 'center', gap: '16px', color: 'var(--text-muted)', textDecoration: 'none' }}>
            <Compass size={22} /> Discover
          </a>
          <a href="#" className="flex items-center gap-4 text-text-muted hover:text-white transition-all" style={{ display: 'flex', alignItems: 'center', gap: '16px', color: 'var(--text-muted)', textDecoration: 'none' }}>
            <Radio size={22} /> Radio
          </a>
          <a href="#" className="flex items-center gap-4 text-text-muted hover:text-white transition-all" style={{ display: 'flex', alignItems: 'center', gap: '16px', color: 'var(--text-muted)', textDecoration: 'none' }}>
            <Mic2 size={22} /> Artists
          </a>
        </nav>

        <div style={{ marginTop: 'auto', paddingTop: '24px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
          <button className="btn-primary w-full" style={{ width: '100%', fontSize: '13px' }} onClick={() => setOnboarded(false)}>Reset Identity</button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        <header className="flex justify-between items-center mb-12" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '48px' }}>
          <div className="relative" style={{ position: 'relative', width: '100%', maxWidth: '480px' }}>
            <Search className="absolute" style={{ position: 'absolute', left: '16px', top: '15px', color: 'var(--text-muted)' }} size={18} />
            <input 
              type="text" 
              placeholder="What are you in the mood for? (Try: Rainy mood rock)" 
              className="search-input"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleSearch}
            />
          </div>
          <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
             <div style={{ padding: '8px 16px', background: 'rgba(255,255,255,0.05)', borderRadius: '30px', fontSize: '12px', border: '1px solid rgba(255,255,255,0.1)' }}>
               Local Node: Active
             </div>
             <div className="w-10 h-10 rounded-full" style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'linear-gradient(45deg, var(--spotify), var(--accent))', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold' }}>T</div>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {!onboarded ? (
            <motion.div 
              key="onboarding"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, y: -20 }}
              style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', textAlign: 'center', maxWidth: '800px', margin: '0 auto' }}
            >
              <h1 className="text-6xl font-extrabold mb-6" style={{ fontSize: '64px', fontWeight: '800', marginBottom: '24px' }}>Welcome to <span className="gradient-text">Discovery</span></h1>
              <p className="text-text-muted text-xl mb-12" style={{ fontSize: '18px', color: 'var(--text-muted)', marginBottom: '48px' }}>Select at least 3 genres to seed your personalized discovery engine.</p>
              
              <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '12px', marginBottom: '48px' }}>
                {topGenres.map(g => (
                  <button 
                    key={g}
                    onClick={() => {
                      if (selectedGenres.includes(g)) setSelectedGenres(prev => prev.filter(x => x !== g));
                      else setSelectedGenres(prev => [...prev, g]);
                    }}
                    style={{
                      padding: '10px 20px',
                      borderRadius: '30px',
                      border: selectedGenres.includes(g) ? '1px solid var(--spotify)' : '1px solid rgba(255,255,255,0.1)',
                      background: selectedGenres.includes(g) ? 'var(--spotify)' : 'transparent',
                      color: selectedGenres.includes(g) ? 'black' : 'var(--text-muted)',
                      fontSize: '14px',
                      fontWeight: selectedGenres.includes(g) ? '600' : '400',
                      cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    {g}
                  </button>
                ))}
              </div>
              <button 
                className="btn-primary" 
                style={{ padding: '16px 48px', fontSize: '18px' }}
                disabled={selectedGenres.length < 3 || loading}
                onClick={startOnboarding}
              >
                {loading ? 'Generating vectors...' : 'Build My Identity'}
              </button>
            </motion.div>
          ) : (
            <motion.div 
              key="feed"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '24px' }}
            >
              <div style={{ gridColumn: '1 / -1', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '16px' }}>
                <div>
                  <h2 style={{ fontSize: '28px', fontWeight: '800', margin: '0' }}>Your Discovery Feed</h2>
                  <p style={{ color: 'var(--text-muted)', margin: '4px 0 0' }}>Ranked for you by Spotify Music Recommender</p>
                </div>
                <button 
                   style={{ background: 'transparent', border: 'none', color: 'var(--spotify)', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }} 
                   onClick={fetchRecommendations}
                >
                  Refresh <SkipForward size={14} />
                </button>
              </div>

              {loading ? (
                Array(12).fill(0).map((_, i) => (
                  <div key={i} className="glass" style={{ aspectRatio: '1/1', animation: 'pulse 2s infinite' }} />
                ))
              ) : (
                recommendations.map(track => (
                  <motion.div 
                    whileHover={{ scale: 1.03, y: -4 }}
                    key={track.track_id} 
                    className="glass"
                    style={{ padding: '16px', cursor: 'pointer', position: 'relative', overflow: 'hidden' }}
                  >
                    <div style={{ aspectRatio: '1/1', background: 'linear-gradient(135deg, rgba(255,255,255,0.1), transparent)', borderRadius: '12px', marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
                      <Music size={42} style={{ opacity: 0.1 }} />
                      {track.discovery_type === 'explore' && (
                        <div style={{ position: 'absolute', top: '8px', right: '8px', background: 'var(--accent)', fontSize: '9px', fontWeight: '800', padding: '3px 8px', borderRadius: '10px' }}>EXPLORE</div>
                      )}
                    </div>
                    <h3 style={{ fontSize: '14px', fontWeight: '700', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', margin: '0' }}>{track.name}</h3>
                    <p style={{ fontSize: '12px', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', margin: '4px 0 12px' }}>{track.artist_name}</p>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ fontSize: '11px', color: 'var(--spotify)', fontWeight: '600' }}>
                        {track.score ? `${Math.round(track.score * 100)}% Match` : 'New Discovery'}
                      </div>
                      <Heart size={14} style={{ color: 'rgba(255,255,255,0.2)' }} />
                    </div>
                  </motion.div>
                ))
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default App;
