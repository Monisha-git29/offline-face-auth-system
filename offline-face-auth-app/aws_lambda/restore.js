const { DynamoDBClient } = require("@aws-sdk/client-dynamodb");
const { DynamoDBDocumentClient, GetCommand } = require("@aws-sdk/lib-dynamodb");

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.DYNAMODB_TABLE || "FaceAuthUsers";

exports.handler = async (event) => {
    console.log("Restore Request Triggered:", JSON.stringify(event));

    try {
        const query = event.queryStringParameters || {};
        const user_id = query.user_id;

        if (!user_id) {
            return {
                statusCode: 400,
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ error: "Missing required query parameter: user_id." })
            };
        }

        const result = await docClient.send(new GetCommand({
            TableName: TABLE_NAME,
            Key: { user_id }
        }));

        if (!result.Item) {
            return {
                statusCode: 404,
                headers: {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                body: JSON.stringify({ error: "User not found in cloud registry." })
            };
        }

        const { name, encrypted_embedding, salt, iv } = result.Item;

        return {
            statusCode: 200,
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            body: JSON.stringify({
                userId: user_id,
                name,
                encryptedEmbedding: encrypted_embedding,
                salt,
                iv
            })
        };
    } catch (error) {
        console.error("Restore query failed:", error);
        return {
            statusCode: 500,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ error: "Internal Server Error: " + error.message })
        };
    }
};
