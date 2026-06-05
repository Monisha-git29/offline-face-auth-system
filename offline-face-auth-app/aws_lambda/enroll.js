const { DynamoDBClient } = require("@aws-sdk/client-dynamodb");
const { DynamoDBDocumentClient, PutCommand } = require("@aws-sdk/lib-dynamodb");

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.DYNAMODB_TABLE || "FaceAuthUsers";

exports.handler = async (event) => {
    console.log("Enrollment Sync Triggered:", JSON.stringify(event));

    try {
        const body = typeof event.body === "string" ? JSON.parse(event.body) : event.body;
        const { user_id, name, encrypted_embedding, salt, iv } = body;

        if (!user_id || !name || !encrypted_embedding || !salt || !iv) {
            return {
                statusCode: 400,
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ error: "Missing required fields in payload." })
            };
        }

        // Put user profile and encrypted embedding to DynamoDB (Enrollment)
        await docClient.send(new PutCommand({
            TableName: TABLE_NAME,
            Item: {
                user_id,
                name,
                encrypted_embedding,
                salt,
                iv,
                created_at: new Date().toISOString(),
                sync_timestamp: Date.now()
            }
        }));

        return {
            statusCode: 201,
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            body: JSON.stringify({ message: "User enrolled successfully in cloud registry.", user_id })
        };
    } catch (error) {
        console.error("Enrollment failed:", error);
        return {
            statusCode: 500,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ error: "Internal Server Error: " + error.message })
        };
    }
};
