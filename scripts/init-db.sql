-- OdorDiff-2 Database Initialization Script

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE odordiff2'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'odordiff2')\gexec

-- Connect to the odordiff2 database
\c odordiff2;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create tables for caching and data management
CREATE TABLE IF NOT EXISTS molecule_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(255) NOT NULL UNIQUE,
    smiles TEXT NOT NULL,
    generation_params JSONB,
    odor_profile JSONB,
    safety_data JSONB,
    synthesis_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ttl_seconds INTEGER DEFAULT 86400,
    tags TEXT[]
);

-- Create table for generation requests
CREATE TABLE IF NOT EXISTS generation_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prompt TEXT NOT NULL,
    num_molecules INTEGER DEFAULT 5,
    parameters JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    result_count INTEGER DEFAULT 0,
    processing_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- Create table for safety assessments
CREATE TABLE IF NOT EXISTS safety_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    smiles TEXT NOT NULL,
    toxicity_score FLOAT,
    skin_sensitizer BOOLEAN,
    eco_score FLOAT,
    ifra_compliant BOOLEAN,
    regulatory_flags JSONB,
    assessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create table for synthesis routes
CREATE TABLE IF NOT EXISTS synthesis_routes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    target_smiles TEXT NOT NULL,
    route_data JSONB NOT NULL,
    feasibility_score FLOAT,
    estimated_cost FLOAT,
    estimated_yield FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create table for user sessions (if authentication is added)
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_token VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255),
    data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Create table for system metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    tags JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_molecule_cache_key ON molecule_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_molecule_cache_created_at ON molecule_cache(created_at);
CREATE INDEX IF NOT EXISTS idx_molecule_cache_accessed_at ON molecule_cache(accessed_at);
CREATE INDEX IF NOT EXISTS idx_molecule_cache_tags ON molecule_cache USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_generation_requests_status ON generation_requests(status);
CREATE INDEX IF NOT EXISTS idx_generation_requests_created_at ON generation_requests(created_at);

CREATE INDEX IF NOT EXISTS idx_safety_assessments_smiles ON safety_assessments(smiles);
CREATE INDEX IF NOT EXISTS idx_safety_assessments_assessed_at ON safety_assessments(assessed_at);

CREATE INDEX IF NOT EXISTS idx_synthesis_routes_target_smiles ON synthesis_routes(target_smiles);
CREATE INDEX IF NOT EXISTS idx_synthesis_routes_feasibility ON synthesis_routes(feasibility_score);

CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON system_metrics(recorded_at);

-- Create views for common queries
CREATE OR REPLACE VIEW recent_generations AS
SELECT 
    id,
    prompt,
    num_molecules,
    status,
    result_count,
    processing_time,
    created_at
FROM generation_requests
WHERE created_at >= NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;

CREATE OR REPLACE VIEW cache_statistics AS
SELECT 
    COUNT(*) as total_entries,
    COUNT(*) FILTER (WHERE accessed_at >= NOW() - INTERVAL '1 hour') as recent_hits,
    AVG(EXTRACT(EPOCH FROM (NOW() - created_at))) as avg_age_seconds,
    SUM(CASE WHEN NOW() > created_at + INTERVAL '1 second' * ttl_seconds THEN 1 ELSE 0 END) as expired_entries
FROM molecule_cache;

-- Create functions for cache management
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM molecule_cache 
    WHERE created_at + INTERVAL '1 second' * ttl_seconds < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION cleanup_old_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions WHERE expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to record metrics
CREATE OR REPLACE FUNCTION record_metric(
    p_metric_name VARCHAR(100),
    p_metric_value FLOAT,
    p_tags JSONB DEFAULT '{}'::jsonb
)
RETURNS UUID AS $$
DECLARE
    metric_id UUID;
BEGIN
    INSERT INTO system_metrics (metric_name, metric_value, tags)
    VALUES (p_metric_name, p_metric_value, p_tags)
    RETURNING id INTO metric_id;
    
    RETURN metric_id;
END;
$$ LANGUAGE plpgsql;

-- Insert initial data
INSERT INTO system_metrics (metric_name, metric_value, tags)
VALUES ('database_initialized', 1.0, '{"version": "1.0.0", "timestamp": "' || NOW()::text || '"}')
ON CONFLICT DO NOTHING;

-- Create cleanup job (requires pg_cron extension in production)
-- SELECT cron.schedule('cleanup-expired-cache', '0 */6 * * *', 'SELECT cleanup_expired_cache();');
-- SELECT cron.schedule('cleanup-old-sessions', '0 2 * * *', 'SELECT cleanup_old_sessions();');

COMMIT;

-- Print initialization summary
SELECT 
    'Database initialized successfully' as status,
    NOW() as timestamp,
    (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public') as tables_created;