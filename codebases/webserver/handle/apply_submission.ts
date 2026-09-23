export default function apply_submission(acc: Site, review: Review): Site {
if (!review.accept || !review.message.trim()) return acc
const entry = { author: review.author.trim().slice(0, 60) || "anonymous", message: review.message.trim().slice(0, 500) }
return { ...acc, entries: acc.entries.concat([entry]).slice(-50) }
}
