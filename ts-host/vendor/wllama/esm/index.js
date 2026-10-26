var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "native_state_paths",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "scor_req": {
    "name": "scor_req",
    "structName": "glue_msg_score_req",
    "className": "GlueMsgScoreReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "scor_res": {
    "name": "scor_res",
    "structName": "glue_msg_score_res",
    "className": "GlueMsgScoreRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "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",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b4-9ff259a";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      if (parsedResult._name === "erro_evt") {
        throw new WllamaRuntimeError(parsedResult.message, "");
      }
      return parsedResult;
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true
    });
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // note: we overlay instead of using llama-server default_template_kwargs, because we cannot transfer complex data structure via GLUE
    // overlay allow mixed data type or nested structure for kwargs
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const sideFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(
              `LoRA adapter ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(
            `LoRA adapter ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      const statePaths = [];
      for (const [i, state] of ((_e = params.native_states) != null ? _e : []).entries()) {
        if (state.blob) {
          if (state.blob.size === 0) {
            throw new WllamaError(
              `Native state ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `native-state-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: state.blob });
          statePaths.push(`models/${name}`);
        } else if (state.path) {
          statePaths.push(state.path);
        } else {
          throw new WllamaError(
            `Native state ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...sideFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_f = params.n_gpu_layers) != null ? _f : 99999,
        n_ctx: (_g = params.n_ctx) != null ? _g : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_h = params.n_parallel) != null ? _h : 4,
        kv_unified: (_i = params.kv_unified) != null ? _i : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_j = params.lora_adapters) == null ? void 0 : _j.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        native_state_paths: params.native_states ? statePaths : void 0,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_k = params.default_template_kwargs) != null ? _k : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  /**
   * Score candidate replies to chat messages (natlang decision readout): the log-probability of each reply as the closed final assistant message, summed over its tokens after the chat prompt. The replies of an item share the prompt.
   * @param options The items to score
   * @returns Per item, the log-probability and token count of each reply, or the error of an item that could not be rendered
   */
  createNatlangScore(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("score", {
        _name: "scor_req",
        data_json: JSON.stringify(options)
      });
      if (!result.success) {
        throw new WllamaError("Model failed to start scoring", "inference_error");
      }
      return yield this.getSingleResult(
        result.req_id,
        "score"
      );
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text" || c.type === "native_state") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
