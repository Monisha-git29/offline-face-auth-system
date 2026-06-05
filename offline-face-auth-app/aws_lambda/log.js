const { DynamoDBClient } = require("@aws-sdk/client-dynamodb");
const { DynamoDBDocumentClient, PutCommand } = require("@aws-sdk/lib-dynamodb");

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.DYNAMODB_LOGS_TABLE || "FaceAuthLogs";

exports.handler = async (event) => {
    console.log("Authentication Log Sync Triggered:", JSON.stringify(event));

    try {
        const body = typeof event.body === "string" ? JSON.parse(event.body) : event.body;
        const { log_id, user_id, similarity_score, liveness_score, status, timestamp } = body;

        if (!log_id || !user_id || similarity_score === undefined || liveness_score === undefined || !status) {
            return {
                statusCode: 400,
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ error: "Missing required fields in log payload." })
            };
        }

        // Put authentication audit log to DynamoDB
        await docClient.send(new PutCommand({
            TableName: TABLE_NAME,
            Item: {
                log_id,
                user_id,
                similarity_score,
                liveness_score,
                status,
                timestamp: timestamp ? new Date(timestamp).toISOString() : new Date().toISOString(),
                sync_timestamp: Date.now()
            }
        }));

        return {
            statusCode: 201,
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            body: JSON.stringify({ message: "Log synced successfully.", log_id })
        };
    } catch (error) {
        console.error("Log upload failed:", error);
        return {
            statusCode: 500,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ error: "Internal Server Error: " + error.message })
        };
    }
};
