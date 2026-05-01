use rusqlite::Connection;

fn main() {
    let conn = Connection::open("C:\\Users\\Yusuf\\agent-rs\\.system_generated\\agent_memory.db").unwrap();
    
    // Check if table exists
    let table_exists: Result<i64, _> = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='cron_jobs'",
        [],
        |row| row.get(0),
    );
    println!("Table cron_jobs exists: {:?}", table_exists);

    // Count active cron jobs
    let active_crons: Result<i64, _> = conn.query_row(
        "SELECT count(*) FROM cron_jobs WHERE completed_at_ms IS NULL",
        [],
        |row| row.get(0),
    );
    println!("Active crons: {:?}", active_crons);

    // Count all cron jobs
    let all_crons: Result<i64, _> = conn.query_row(
        "SELECT count(*) FROM cron_jobs",
        [],
        |row| row.get(0),
    );
    println!("All crons: {:?}", all_crons);
}
